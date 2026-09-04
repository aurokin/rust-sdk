//! A receiver SHOULD NOT send a response for a request it has already been told
//! to cancel. This drives a real stdio server with raw JSON-RPC: the tool blocks
//! until the request is cancelled, so its result is only produced *after* the
//! cancellation — the service loop must drop it rather than write it to the wire.

#[cfg(all(feature = "client", not(feature = "local")))]
use std::sync::Arc;
#[cfg(all(feature = "client", not(feature = "local")))]
use std::sync::atomic::{AtomicUsize, Ordering};
use std::{collections::BTreeSet, process::Stdio, time::Duration};

#[cfg(all(feature = "client", not(feature = "local")))]
use rmcp::{
    ClientHandler, RoleClient,
    model::{
        CancelledNotification, CancelledNotificationParam, ClientJsonRpcMessage, ClientRequest,
        ClientResult, ElicitRequest, ElicitRequestParams, ElicitResult, ElicitationAction,
        ElicitationSchema, PingRequest, RequestId, ServerJsonRpcMessage, ServerNotification,
        ServerRequest, ServerResult,
    },
    service::{QuitReason, serve_directly},
    transport::{IntoTransport, Transport},
};
use rmcp::{
    ErrorData as McpError, RoleServer, ServerHandler, ServiceExt,
    model::{
        CallToolRequestParams, CallToolResponse, CallToolResult, ContentBlock, ServerCapabilities,
        ServerInfo,
    },
    service::RequestContext,
};
use serde_json::{Value, json};
use tokio::{
    io::{AsyncBufReadExt, AsyncWrite, AsyncWriteExt, BufReader},
    process::{Child, Command},
};

const HELPER_ENV: &str = "RMCP_CANCELLED_RESPONSE_HELPER";
const READ_TIMEOUT: Duration = Duration::from_secs(10);
const COMPUTER_USE_REQUEST_ID: &str = "computer-use-request-2";

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cancelled_request_receives_no_response() -> anyhow::Result<()> {
    let mut child = spawn_helper();
    let mut writer = child.stdin.take().expect("helper stdin");
    let stdout = child.stdout.take().expect("helper stdout");
    let mut reader = BufReader::new(stdout);

    send_json(
        &mut writer,
        &json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": { "name": "raw-test-client", "version": "0.0.0" }
            }
        }),
    )
    .await?;
    collect_ids_until(&mut reader, "1", READ_TIMEOUT).await?;
    send_json(
        &mut writer,
        &json!({ "jsonrpc": "2.0", "method": "notifications/initialized" }),
    )
    .await?;

    // Start a request that blocks until cancelled, then cancel it. Its response is
    // produced only after the cancellation arrives, so it must be suppressed.
    send_json(
        &mut writer,
        &json!({
            "jsonrpc": "2.0",
            "id": COMPUTER_USE_REQUEST_ID,
            "method": "tools/call",
            "params": { "name": "wait-for-cancel", "arguments": {} }
        }),
    )
    .await?;
    send_json(
        &mut writer,
        &json!({
            "jsonrpc": "2.0",
            "method": "notifications/cancelled",
            "params": { "requestId": "unrelated-request" }
        }),
    )
    .await?;
    send_json(
        &mut writer,
        &json!({
            "jsonrpc": "2.0",
            "method": "notifications/cancelled",
            "params": { "requestId": COMPUTER_USE_REQUEST_ID }
        }),
    )
    .await?;
    // A ping proves the server is alive past the cancellation, so the absence of
    // a cancelled response is genuine suppression rather than a dead connection.
    send_json(
        &mut writer,
        &json!({ "jsonrpc": "2.0", "id": 3, "method": "ping" }),
    )
    .await?;

    let seen = collect_ids_until(&mut reader, "3", READ_TIMEOUT).await?;
    assert!(seen.contains("3"));
    assert!(!seen.contains(COMPUTER_USE_REQUEST_ID));

    send_json(
        &mut writer,
        &json!({
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": { "name": "complete-unless-cancelled", "arguments": {} }
        }),
    )
    .await?;
    send_json(
        &mut writer,
        &json!({
            "jsonrpc": "2.0",
            "method": "notifications/cancelled",
            "params": { "requestId": "4" }
        }),
    )
    .await?;
    send_json(
        &mut writer,
        &json!({ "jsonrpc": "2.0", "id": 5, "method": "ping" }),
    )
    .await?;

    let seen = collect_ids_until(&mut reader, "5", READ_TIMEOUT).await?;
    assert!(seen.contains("4"), "string and numeric IDs must not alias");
    assert!(seen.contains("5"));

    drop(writer);
    wait_for_child(&mut child).await;
    Ok(())
}

struct WaitForCancelServer;

impl ServerHandler for WaitForCancelServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        context: RequestContext<RoleServer>,
    ) -> Result<CallToolResponse, McpError> {
        if request.name == "complete-unless-cancelled" {
            tokio::select! {
                _ = context.ct.cancelled() => {}
                _ = tokio::time::sleep(Duration::from_millis(100)) => {}
            }
            return Ok(CallToolResult::success(vec![ContentBlock::text("completed")]).into());
        }
        context.ct.cancelled().await;
        Ok(CallToolResult::success(vec![ContentBlock::text("late response")]).into())
    }
}

#[cfg(all(feature = "client", not(feature = "local")))]
#[derive(Clone)]
struct WaitForReverseCancelClient {
    events: tokio::sync::mpsc::UnboundedSender<&'static str>,
    finish: Option<Arc<tokio::sync::Notify>>,
}

#[cfg(all(feature = "client", not(feature = "local")))]
impl ClientHandler for WaitForReverseCancelClient {
    async fn create_elicitation(
        &self,
        _request: ElicitRequestParams,
        context: rmcp::service::RequestContext<RoleClient>,
    ) -> Result<ElicitResult, McpError> {
        self.events.send("started").expect("test is listening");
        context.ct.cancelled().await;
        self.events.send("cancelled").expect("test is listening");
        if let Some(finish) = &self.finish {
            finish.notified().await;
        }
        Ok(ElicitResult::new(ElicitationAction::Decline))
    }
}

#[cfg(all(feature = "client", not(feature = "local")))]
#[tokio::test]
async fn peer_cancels_reverse_request_without_cancelling_outbound_request() -> anyhow::Result<()> {
    const REVERSE_REQUEST_ID: &str = "computer-use-elicitation-1";
    const AFTER_CANCEL_ID: &str = "after-cancel";

    let (client_transport, server_transport) = tokio::io::duplex(4096);
    let (events_tx, mut events_rx) = tokio::sync::mpsc::unbounded_channel();
    let client = serve_directly::<RoleClient, _, _, _, _>(
        WaitForReverseCancelClient {
            events: events_tx,
            finish: None,
        },
        client_transport,
        None,
    );
    let mut server = IntoTransport::<RoleServer, _, _>::into_transport(server_transport);

    let outbound = tokio::spawn({
        let peer = client.peer().clone();
        async move {
            peer.send_request(ClientRequest::PingRequest(PingRequest {
                method: Default::default(),
                extensions: Default::default(),
            }))
            .await
        }
    });
    let Some(ClientJsonRpcMessage::Request(outbound_request)) = server.receive().await else {
        panic!("expected outbound ping request");
    };

    server
        .send(ServerJsonRpcMessage::request(
            ServerRequest::ElicitRequest(ElicitRequest::new(
                ElicitRequestParams::FormElicitationParams {
                    meta: None,
                    message: "Continue using computer control?".to_owned(),
                    requested_schema: ElicitationSchema::builder()
                        .build()
                        .expect("empty elicitation schema is valid"),
                },
            )),
            RequestId::String(REVERSE_REQUEST_ID.into()),
        ))
        .await?;
    assert_eq!(
        tokio::time::timeout(READ_TIMEOUT, events_rx.recv()).await?,
        Some("started")
    );

    server
        .send(ServerJsonRpcMessage::notification(
            ServerNotification::CancelledNotification(CancelledNotification::new(
                CancelledNotificationParam::new(
                    Some(outbound_request.id.clone()),
                    Some("unrelated direction".to_owned()),
                ),
            )),
        ))
        .await?;
    assert!(
        tokio::time::timeout(Duration::from_millis(100), events_rx.recv())
            .await
            .is_err(),
        "unrelated cancellation must not cancel the reverse request"
    );
    assert!(
        !outbound.is_finished(),
        "peer cancellation must not cancel an outbound request with the same ID"
    );

    server
        .send(ServerJsonRpcMessage::response(
            ServerResult::empty(()),
            outbound_request.id,
        ))
        .await?;
    assert!(matches!(outbound.await??, ServerResult::EmptyResult(_)));

    server
        .send(ServerJsonRpcMessage::notification(
            ServerNotification::CancelledNotification(CancelledNotification::new(
                CancelledNotificationParam::new(
                    Some(RequestId::String(REVERSE_REQUEST_ID.into())),
                    Some("user cancelled".to_owned()),
                ),
            )),
        ))
        .await?;
    assert_eq!(
        tokio::time::timeout(READ_TIMEOUT, events_rx.recv()).await?,
        Some("cancelled"),
        "the matching cancellation must fire RequestContext.ct"
    );

    server
        .send(ServerJsonRpcMessage::request(
            ServerRequest::PingRequest(PingRequest {
                method: Default::default(),
                extensions: Default::default(),
            }),
            RequestId::String(AFTER_CANCEL_ID.into()),
        ))
        .await?;
    let Some(ClientJsonRpcMessage::Response(response)) =
        tokio::time::timeout(READ_TIMEOUT, server.receive()).await?
    else {
        panic!("expected ping response after cancellation");
    };
    assert_eq!(response.id, RequestId::String(AFTER_CANCEL_ID.into()));
    assert!(matches!(response.result, ClientResult::EmptyResult(_)));
    assert!(
        tokio::time::timeout(Duration::from_millis(300), server.receive())
            .await
            .is_err(),
        "cancelled reverse request must not send a late response"
    );

    client.cancel().await?;
    Ok(())
}

#[cfg(all(feature = "client", not(feature = "local")))]
struct ChannelClientTransport {
    incoming: tokio::sync::mpsc::UnboundedReceiver<ServerJsonRpcMessage>,
    outgoing: tokio::sync::mpsc::UnboundedSender<ClientJsonRpcMessage>,
    eof_observed: Arc<tokio::sync::Notify>,
}

#[cfg(all(feature = "client", not(feature = "local")))]
impl Transport<RoleClient> for ChannelClientTransport {
    type Error = std::io::Error;

    fn send(
        &mut self,
        item: ClientJsonRpcMessage,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send + 'static {
        let outgoing = self.outgoing.clone();
        async move {
            outgoing.send(item).map_err(|_| {
                std::io::Error::new(std::io::ErrorKind::BrokenPipe, "test receiver closed")
            })
        }
    }

    async fn receive(&mut self) -> Option<ServerJsonRpcMessage> {
        let message = self.incoming.recv().await;
        if message.is_none() {
            self.eof_observed.notify_one();
        }
        message
    }

    async fn close(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }
}

#[cfg(all(feature = "client", not(feature = "local")))]
#[derive(Clone)]
struct ReusedIdClient {
    calls: Arc<AtomicUsize>,
    events: tokio::sync::mpsc::UnboundedSender<&'static str>,
    first_finish: Arc<tokio::sync::Notify>,
    second_finish: Arc<tokio::sync::Notify>,
}

#[cfg(all(feature = "client", not(feature = "local")))]
impl ClientHandler for ReusedIdClient {
    async fn create_elicitation(
        &self,
        _request: ElicitRequestParams,
        context: rmcp::service::RequestContext<RoleClient>,
    ) -> Result<ElicitResult, McpError> {
        match self.calls.fetch_add(1, Ordering::SeqCst) {
            0 => {
                self.events
                    .send("first-started")
                    .expect("test is listening");
                context.ct.cancelled().await;
                self.events
                    .send("first-cancelled")
                    .expect("test is listening");
                self.first_finish.notified().await;
                Ok(ElicitResult::new(ElicitationAction::Decline))
            }
            1 => {
                self.events
                    .send("second-started")
                    .expect("test is listening");
                self.second_finish.notified().await;
                Ok(ElicitResult::new(ElicitationAction::Accept))
            }
            call => panic!("unexpected elicitation call {call}"),
        }
    }
}

#[cfg(all(feature = "client", not(feature = "local")))]
fn reverse_elicitation(id: &str) -> ServerJsonRpcMessage {
    ServerJsonRpcMessage::request(
        ServerRequest::ElicitRequest(ElicitRequest::new(
            ElicitRequestParams::FormElicitationParams {
                meta: None,
                message: "Continue using computer control?".to_owned(),
                requested_schema: ElicitationSchema::builder()
                    .build()
                    .expect("empty elicitation schema is valid"),
            },
        )),
        RequestId::String(id.into()),
    )
}

#[cfg(all(feature = "client", not(feature = "local")))]
#[tokio::test]
async fn cancelled_response_cannot_consume_reused_request_id() -> anyhow::Result<()> {
    const REUSED_REQUEST_ID: &str = "computer-use-reused-id";
    const AFTER_REUSE_ID: &str = "after-reuse";

    let (to_client, incoming) = tokio::sync::mpsc::unbounded_channel();
    let (outgoing, mut from_client) = tokio::sync::mpsc::unbounded_channel();
    let eof_observed = Arc::new(tokio::sync::Notify::new());
    let first_finish = Arc::new(tokio::sync::Notify::new());
    let second_finish = Arc::new(tokio::sync::Notify::new());
    let (events, mut event_rx) = tokio::sync::mpsc::unbounded_channel();
    let client = serve_directly::<RoleClient, _, _, _, _>(
        ReusedIdClient {
            calls: Arc::new(AtomicUsize::new(0)),
            events,
            first_finish: first_finish.clone(),
            second_finish: second_finish.clone(),
        },
        ChannelClientTransport {
            incoming,
            outgoing,
            eof_observed,
        },
        None,
    );

    to_client.send(reverse_elicitation(REUSED_REQUEST_ID))?;
    assert_eq!(event_rx.recv().await, Some("first-started"));
    to_client.send(ServerJsonRpcMessage::notification(
        ServerNotification::CancelledNotification(CancelledNotification::new(
            CancelledNotificationParam::new(
                Some(RequestId::String(REUSED_REQUEST_ID.into())),
                Some("user cancelled".to_owned()),
            ),
        )),
    ))?;
    assert_eq!(event_rx.recv().await, Some("first-cancelled"));

    to_client.send(reverse_elicitation(REUSED_REQUEST_ID))?;
    assert_eq!(event_rx.recv().await, Some("second-started"));
    first_finish.notify_one();
    assert!(
        tokio::time::timeout(Duration::from_millis(300), from_client.recv())
            .await
            .is_err(),
        "the cancelled handler's stale response must remain suppressed"
    );

    second_finish.notify_one();
    let Some(ClientJsonRpcMessage::Response(response)) =
        tokio::time::timeout(READ_TIMEOUT, from_client.recv()).await?
    else {
        panic!("expected response from the replacement request");
    };
    assert_eq!(response.id, RequestId::String(REUSED_REQUEST_ID.into()));
    let ClientResult::ElicitResult(result) = response.result else {
        panic!("expected elicitation response");
    };
    assert_eq!(result.action, ElicitationAction::Accept);

    to_client.send(ServerJsonRpcMessage::request(
        ServerRequest::PingRequest(PingRequest {
            method: Default::default(),
            extensions: Default::default(),
        }),
        RequestId::String(AFTER_REUSE_ID.into()),
    ))?;
    let Some(ClientJsonRpcMessage::Response(response)) =
        tokio::time::timeout(READ_TIMEOUT, from_client.recv()).await?
    else {
        panic!("expected ping response after ID reuse");
    };
    assert_eq!(response.id, RequestId::String(AFTER_REUSE_ID.into()));

    drop(to_client);
    assert!(matches!(
        tokio::time::timeout(READ_TIMEOUT, client.waiting()).await??,
        QuitReason::Closed
    ));
    Ok(())
}

#[cfg(all(feature = "client", not(feature = "local")))]
#[tokio::test]
async fn graceful_close_drains_uncancelled_reverse_response() -> anyhow::Result<()> {
    const REVERSE_REQUEST_ID: &str = "computer-use-close-drain";

    let (to_client, incoming) = tokio::sync::mpsc::unbounded_channel();
    let (outgoing, mut from_client) = tokio::sync::mpsc::unbounded_channel();
    let eof_observed = Arc::new(tokio::sync::Notify::new());
    let finish = Arc::new(tokio::sync::Notify::new());
    let (events_tx, mut events_rx) = tokio::sync::mpsc::unbounded_channel();
    let mut client = serve_directly::<RoleClient, _, _, _, _>(
        WaitForReverseCancelClient {
            events: events_tx,
            finish: Some(finish.clone()),
        },
        ChannelClientTransport {
            incoming,
            outgoing,
            eof_observed,
        },
        None,
    );

    to_client.send(reverse_elicitation(REVERSE_REQUEST_ID))?;
    assert_eq!(events_rx.recv().await, Some("started"));

    let (close_result, cancellation_event) = tokio::join!(client.close(), async {
        let event = events_rx.recv().await;
        finish.notify_one();
        event
    });
    assert!(matches!(close_result?, QuitReason::Cancelled));
    assert_eq!(cancellation_event, Some("cancelled"));

    let Some(ClientJsonRpcMessage::Response(response)) =
        tokio::time::timeout(READ_TIMEOUT, from_client.recv()).await?
    else {
        panic!("expected in-flight response during graceful close");
    };
    assert_eq!(response.id, RequestId::String(REVERSE_REQUEST_ID.into()));
    Ok(())
}

#[cfg(all(feature = "client", not(feature = "local")))]
#[tokio::test]
async fn cancelled_reverse_request_stays_suppressed_during_eof_drain() -> anyhow::Result<()> {
    const REVERSE_REQUEST_ID: &str = "computer-use-elicitation-before-eof";

    let (to_client, incoming) = tokio::sync::mpsc::unbounded_channel();
    let (outgoing, mut from_client) = tokio::sync::mpsc::unbounded_channel();
    let eof_observed = Arc::new(tokio::sync::Notify::new());
    let finish = Arc::new(tokio::sync::Notify::new());
    let (events_tx, mut events_rx) = tokio::sync::mpsc::unbounded_channel();
    let client = serve_directly::<RoleClient, _, _, _, _>(
        WaitForReverseCancelClient {
            events: events_tx,
            finish: Some(finish.clone()),
        },
        ChannelClientTransport {
            incoming,
            outgoing,
            eof_observed: eof_observed.clone(),
        },
        None,
    );

    to_client.send(reverse_elicitation(REVERSE_REQUEST_ID))?;
    assert_eq!(
        tokio::time::timeout(READ_TIMEOUT, events_rx.recv()).await?,
        Some("started")
    );

    to_client.send(ServerJsonRpcMessage::notification(
        ServerNotification::CancelledNotification(CancelledNotification::new(
            CancelledNotificationParam::new(
                Some(RequestId::String(REVERSE_REQUEST_ID.into())),
                Some("user cancelled".to_owned()),
            ),
        )),
    ))?;
    assert_eq!(
        tokio::time::timeout(READ_TIMEOUT, events_rx.recv()).await?,
        Some("cancelled")
    );

    let eof = eof_observed.notified();
    drop(to_client);
    tokio::time::timeout(READ_TIMEOUT, eof).await?;
    finish.notify_one();

    assert!(
        matches!(
            tokio::time::timeout(READ_TIMEOUT, client.waiting()).await??,
            QuitReason::Closed
        ),
        "client must close after input EOF"
    );
    assert!(
        from_client.recv().await.is_none(),
        "cancelled reverse request must not be sent during EOF drain"
    );
    Ok(())
}

#[tokio::test]
async fn cancelled_response_helper() -> anyhow::Result<()> {
    if std::env::var(HELPER_ENV).as_deref() != Ok("1") {
        return Ok(());
    }
    run_helper_server().await?;
    Ok(())
}

#[cfg(feature = "local")]
async fn run_helper_server() -> anyhow::Result<()> {
    tokio::task::LocalSet::new()
        .run_until(serve_helper_stdio())
        .await
}

#[cfg(not(feature = "local"))]
async fn run_helper_server() -> anyhow::Result<()> {
    serve_helper_stdio().await
}

async fn serve_helper_stdio() -> anyhow::Result<()> {
    let server = WaitForCancelServer.serve(rmcp::transport::stdio()).await?;
    server.waiting().await?;
    Ok(())
}

fn spawn_helper() -> Child {
    let exe = std::env::current_exe().expect("current test exe");
    Command::new(exe)
        .arg("--exact")
        .arg("cancelled_response_helper")
        .arg("--quiet")
        .arg("--nocapture")
        .arg("--test-threads")
        .arg("1")
        .env(HELPER_ENV, "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .kill_on_drop(true)
        .spawn()
        .expect("spawn helper")
}

async fn wait_for_child(child: &mut Child) {
    let _ = tokio::time::timeout(Duration::from_secs(2), child.wait()).await;
    if child.id().is_some() {
        let _ = child.kill().await;
    }
}

async fn send_json<W>(writer: &mut W, message: &Value) -> anyhow::Result<()>
where
    W: AsyncWrite + Unpin,
{
    let serialized = serde_json::to_string(message)?;
    writer.write_all(serialized.as_bytes()).await?;
    writer.write_all(b"\n").await?;
    writer.flush().await?;
    Ok(())
}

/// Read response lines, collecting every message id seen, until `stop_id` is seen
/// (then a short grace read to catch any straggler) or the timeout elapses.
async fn collect_ids_until<R>(
    reader: &mut BufReader<R>,
    stop_id: &str,
    timeout: Duration,
) -> anyhow::Result<BTreeSet<String>>
where
    R: tokio::io::AsyncRead + Unpin,
{
    let mut seen = BTreeSet::new();
    let mut deadline = tokio::time::Instant::now() + timeout;
    loop {
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        if remaining.is_zero() {
            break;
        }
        let mut line = String::new();
        let Ok(read_result) = tokio::time::timeout(remaining, reader.read_line(&mut line)).await
        else {
            break;
        };
        if read_result? == 0 {
            break;
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let Ok(value) = serde_json::from_str::<Value>(trimmed) else {
            continue;
        };
        let id = match value.get("id") {
            Some(Value::String(id)) => Some(id.clone()),
            Some(Value::Number(id)) => Some(id.to_string()),
            _ => None,
        };
        if let Some(id) = id {
            seen.insert(id.clone());
            if id == stop_id {
                // Give any late (incorrectly-sent) response a brief window to arrive.
                deadline = tokio::time::Instant::now() + Duration::from_millis(300);
            }
        }
    }
    Ok(seen)
}
