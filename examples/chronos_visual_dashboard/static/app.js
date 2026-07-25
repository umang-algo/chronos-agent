/**
 * Chronos-Agent Interactive Substrate Visualizer Client Logic
 */

document.addEventListener("DOMContentLoaded", () => {

  // Initialize Mermaid
  mermaid.initialize({
    startOnLoad: false,
    theme: 'dark',
    securityLevel: 'loose',
    flowchart: { curve: 'basis' }
  });

  // DOM References
  const valStep = document.getElementById("valStep");
  const valCheckpoints = document.getElementById("valCheckpoints");
  const valNodes = document.getElementById("valNodes");
  const valRollbacks = document.getElementById("valRollbacks");
  const modelName = document.getElementById("modelName");
  const cpBadge = document.getElementById("cpBadge");

  const chatStream = document.getElementById("chatStream");
  const chatForm = document.getElementById("chatForm");
  const chatInput = document.getElementById("chatInput");
  const btnSend = document.getElementById("btnSend");

  const quickFileViewer = document.getElementById("quickFileViewer");
  const fullFileViewer = document.getElementById("fullFileViewer");
  const quickHintBox = document.getElementById("quickHintBox");
  const dagFailureExplanation = document.getElementById("dagFailureExplanation");
  const mermaidGraph = document.getElementById("mermaidGraph");

  const checkpointsGrid = document.getElementById("checkpointsGrid");
  const fullFileTree = document.getElementById("fullFileTree");
  const logStream = document.getElementById("logStream");

  const btnReset = document.getElementById("btnReset");
  const btnClearLogs = document.getElementById("btnClearLogs");
  const btnQuickRefreshFile = document.getElementById("btnQuickRefreshFile");
  const btnRedrawDAG = document.getElementById("btnRedrawDAG");

  let activeFile = "mock_db/financial_ledger.json";

  // Tab Switcher Logic
  document.querySelectorAll(".tab-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
      document.querySelectorAll(".tab-pane").forEach(p => p.classList.remove("active"));

      btn.classList.add("active");
      const targetPane = document.getElementById(btn.dataset.tab);
      if (targetPane) targetPane.classList.add("active");

      // Auto refresh on tab switch
      refreshDashboard();
      if (btn.dataset.tab === "tab-files") loadWorkspaceFiles();
    });
  });

  // Quick Prompt Pills
  document.querySelectorAll(".prompt-pill").forEach(pill => {
    pill.addEventListener("click", () => {
      chatInput.value = pill.dataset.prompt;
      chatInput.focus();
    });
  });

  // Initial Sync
  refreshDashboard();
  loadWorkspaceFiles();
  loadSelectedFileContent(activeFile);

  // Auto Polling Sync Loop (Every 3 seconds)
  setInterval(() => {
    refreshDashboard();
  }, 3000);

  // Chat Submission Handler
  chatForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const promptText = chatInput.value.trim();
    if (!promptText) return;

    chatInput.value = "";
    btnSend.disabled = true;

    appendChatMessage("user", "👤 User", promptText);
    logMessage("info", `Prompt Sent: "${promptText}"`);

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: promptText })
      });

      const res = await response.json();

      if (res.events && res.events.length > 0) {
        for (const evt of res.events) {
          if (evt.type === "tool_call") {
            appendChatMessage("alert", "⚙️ Tool Call Intent", `Tool: ${evt.name} | Args: ${JSON.stringify(evt.args)}`);
          } else if (evt.type === "tool_result") {
            if (evt.success) {
              logMessage("success", `[Chronos Commit] Tool '${evt.name}' executed cleanly.`);
            } else {
              appendChatMessage("alert", "🛡️ Chronos Guard Interception", `REJECTED: ${evt.hint}`);
              logMessage("warning", `[Chronos Guard] Intercepted/Reverted: ${evt.hint}`);
            }
          }
        }
      }

      appendChatMessage("assistant", "🤖 Agent", res.assistant_response);

      // Instant UI Sync
      refreshDashboard();
      loadSelectedFileContent(activeFile);

    } catch (err) {
      appendChatMessage("alert", "❌ Error", err.message);
      logMessage("error", `Execution Error: ${err.message}`);
    } finally {
      btnSend.disabled = false;
    }
  });

  // Reset & Refresh Buttons
  btnReset.addEventListener("click", async () => {
    logMessage("info", "Resetting workspace to clean 1,000 orders state...");
    await fetch("/api/reset");
    chatStream.innerHTML = `<div class="chat-msg msg-system"><span class="msg-author">🤖 Chronos System</span><div class="msg-body">Workspace reset to 1,000 orders clean state. Checkpoints cleared.</div></div>`;
    refreshDashboard();
    loadWorkspaceFiles();
    loadSelectedFileContent("mock_db/financial_ledger.json");
  });

  btnClearLogs.addEventListener("click", () => { logStream.innerHTML = ""; });
  btnQuickRefreshFile.addEventListener("click", () => loadSelectedFileContent(activeFile));
  btnRedrawDAG.addEventListener("click", () => refreshDashboard(true));

  // -------------------------------------------------------------------------
  // Dashboard Sync Pipeline
  // -------------------------------------------------------------------------

  async function refreshDashboard(forceDAGRedraw = false) {
    try {
      // 1. System Status
      const resStatus = await fetch("/api/status");
      const status = await resStatus.json();

      valStep.textContent = status.current_step;
      valCheckpoints.textContent = status.checkpoints_count;
      valNodes.textContent = status.ctg_nodes_count;
      valRollbacks.textContent = status.failures_count;
      cpBadge.textContent = `${status.checkpoints_count} Checkpoints`;

      if (status.model_name) modelName.textContent = status.model_name;

      // 2. Checkpoints Grid
      const resCP = await fetch("/api/checkpoints");
      const cpData = await resCP.json();
      renderCheckpointsGrid(cpData.checkpoints || []);

      // 3. CTG DAG Data
      const resDAG = await fetch("/api/ctg-dag");
      const dagData = await resDAG.json();

      const explanation = dagData.latest_failure_explanation || "✅ Pre-flight guards active. Zero unhandled failure state on disk.";
      quickHintBox.textContent = explanation;
      dagFailureExplanation.textContent = explanation;

      if (dagData.mermaid) {
        renderMermaidGraph(dagData.mermaid);
      }

    } catch (err) {
      console.warn("Sync warning:", err);
    }
  }

  // -------------------------------------------------------------------------
  // Renderers
  // -------------------------------------------------------------------------

  function renderCheckpointsGrid(checkpoints) {
    if (!checkpoints || checkpoints.length === 0) {
      checkpointsGrid.innerHTML = `<div class="cp-empty">No checkpoints recorded yet. Run an action to record snapshots!</div>`;
      return;
    }

    checkpointsGrid.innerHTML = checkpoints.map(cp => {
      const timeStr = new Date(cp.created_at * 1000).toLocaleTimeString();
      return `
        <div class="cp-card">
          <div class="flex-between">
            <span class="cp-title">Step ${cp.step_number}</span>
            <span class="badge badge-purple">${timeStr}</span>
          </div>
          <div class="cp-meta">CP ID: ${cp.checkpoint_id}</div>
          <div class="cp-meta">Snapshot: ${cp.fs_snapshot_id}</div>
          <div class="cp-meta">Prompts: ${cp.prompt_snapshot_count} msgs</div>
        </div>
      `;
    }).join("");
  }

  async function loadWorkspaceFiles() {
    try {
      const res = await fetch("/api/files");
      const data = await res.json();
      const files = data.files || [];

      fullFileTree.innerHTML = files.map(f => {
        const activeClass = f.rel_path === activeFile ? "active" : "";
        return `<div class="tree-node ${activeClass}" data-path="${f.rel_path}">${f.rel_path} (${(f.size_bytes / 1024).toFixed(1)}KB)</div>`;
      }).join("");

      fullFileTree.querySelectorAll(".tree-node").forEach(item => {
        item.addEventListener("click", () => {
          fullFileTree.querySelectorAll(".tree-node").forEach(i => i.classList.remove("active"));
          item.classList.add("active");
          activeFile = item.dataset.path;
          loadSelectedFileContent(activeFile);
        });
      });
    } catch (err) {
      console.error("File tree load error:", err);
    }
  }

  async function loadSelectedFileContent(relPath) {
    try {
      const res = await fetch("/api/file-content", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ rel_path: relPath })
      });
      const data = await res.json();
      if (data.success) {
        quickFileViewer.textContent = data.content;
        fullFileViewer.textContent = data.content;
      } else {
        quickFileViewer.textContent = `Error: ${data.error}`;
        fullFileViewer.textContent = `Error: ${data.error}`;
      }
    } catch (err) {
      quickFileViewer.textContent = `Error: ${err.message}`;
    }
  }

  async function renderMermaidGraph(mermaidCode) {
    try {
      mermaidGraph.removeAttribute("data-processed");
      mermaidGraph.textContent = mermaidCode;
      await mermaid.run({ nodes: [mermaidGraph] });
    } catch (err) {
      console.warn("Mermaid render note:", err);
    }
  }

  function appendChatMessage(role, author, text) {
    const msg = document.createElement("div");
    msg.className = `chat-msg msg-${role}`;
    msg.innerHTML = `<span class="msg-author">${escapeHtml(author)}</span><div class="msg-body">${escapeHtml(text)}</div>`;
    chatStream.appendChild(msg);
    chatStream.scrollTop = chatStream.scrollHeight;
  }

  function logMessage(type, message) {
    const entry = document.createElement("div");
    entry.className = `log-entry log-${type}`;
    const timeStr = new Date().toLocaleTimeString();
    entry.innerHTML = `<span class="log-time">[${timeStr}]</span> <span class="log-msg">${escapeHtml(message)}</span>`;
    logStream.appendChild(entry);
    logStream.scrollTop = logStream.scrollHeight;
  }

  function escapeHtml(str) {
    return String(str).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

});
