const bridge = window.AstrBotPluginView || window.AstrBotPluginPage;
const $ = (id) => document.getElementById(id);
const state = {
  tab: "history",
  page: 1,
  userPage: 1,
  items: [],
  selected: new Set(),
  epoch: 0,
  settings: null,
  dirty: false,
};
const cache = new Map();
const date = (value) =>
  new Intl.DateTimeFormat("zh-CN", {
    timeZone: "Asia/Shanghai",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  }).format(new Date(value * 1000));
const el = (tag, cls, text) => {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
};
const button = (text, cls, action) => {
  const node = el("button", cls, text);
  node.type = "button";
  node.addEventListener("click", () => run(action, node));
  return node;
};
const api = (endpoint, params) => bridge.apiGet(`admin/${endpoint}`, params);
const post = (endpoint, body) => bridge.apiPost(`admin/${endpoint}`, body);
function notice(message, error = false) {
  $("notice").textContent = message;
  $("notice").className = error ? "error" : "";
  $("notice").hidden = !message;
  document.querySelector(".dialog-error")?.remove();
  if (error && $("detail-dialog").open) {
    const feedback = el("p", "dialog-error", message);
    feedback.setAttribute("role", "alert");
    $("detail-body").prepend(feedback);
  }
}
async function run(action, node) {
  if (node) node.disabled = true;
  try {
    await action();
  } catch (error) {
    notice(error.message || "操作失败，请重试。", true);
  } finally {
    if (node?.isConnected) node.disabled = false;
  }
}
function empty(target, text) {
  target.replaceChildren(el("div", "empty", text));
}
function query() {
  return Object.fromEntries(new FormData($("filters")));
}
function renderStats(stats) {
  $("stats").replaceChildren(
    ...[
      ["成功归档", stats.total],
      ["生图用户", stats.users],
      ["最新图库编号", stats.latest_id ? `#${stats.latest_id}` : "—"],
      ["回收站", stats.deleted],
    ].map(([label, value]) => {
      const card = el("div", "stat");
      card.append(el("span", "", label), el("strong", "", value));
      return card;
    }),
  );
}
function pagination(target, data, load) {
  const pages = Math.max(1, Math.ceil(data.total / data.limit));
  const previous = button("← 上一页", "secondary", () => load(data.page - 1));
  previous.disabled = data.page <= 1;
  const next = button("下一页 →", "secondary", () => load(data.page + 1));
  next.disabled = data.page >= pages;
  target.replaceChildren(
    previous,
    el("span", "", `第 ${data.page} / ${pages} 页 · 共 ${data.total} 项`),
    next,
  );
}
async function preview(record, target, large = false) {
  const key = `${record.image_id}:${record.revision}:${large}`;
  try {
    let data = cache.get(key);
    if (!data) {
      data = await api("preview", {
        image_id: record.image_id,
        large: large ? "1" : "0",
      });
      cache.set(key, data);
      if (cache.size > 120) cache.delete(cache.keys().next().value);
    }
    if (!target.isConnected) return;
    const img = el("img");
    img.alt = `图库 #${record.image_id}`;
    img.src = data.data_url;
    target.replaceChildren(img);
  } catch (error) {
    if (target.isConnected)
      target.textContent = error.message || "无法读取图片";
  }
}
function selection() {
  $("batch-action").disabled = !state.selected.size;
  $("batch-action").textContent =
    `${query().deleted === "1" ? "恢复所选" : "移入回收站"}${state.selected.size ? `（${state.selected.size}）` : ""}`;
  $("select-all").checked =
    !!state.items.length && state.selected.size === state.items.length;
  $("select-all").indeterminate =
    state.selected.size > 0 && state.selected.size < state.items.length;
}
async function loadHistory(page = 1) {
  state.page = page;
  const epoch = ++state.epoch;
  state.selected.clear();
  state.items = [];
  selection();
  empty($("image-list"), "正在读取图库…");
  $("image-pagination").replaceChildren();
  const data = await api("images", { ...query(), page, limit: 24 });
  if (epoch !== state.epoch) return;
  if (page > 1 && !data.items.length) return loadHistory(page - 1);
  state.items = data.items;
  renderStats(data.stats);
  $("result-count").textContent = `共 ${data.total} 张图片`;
  $("image-list").replaceChildren();
  if (!data.items.length)
    empty($("image-list"), "没有匹配的图片。试试其他关键词，或调整筛选条件。");
  const covers = [];
  for (const record of data.items) {
    const card = el("article", "image-card");
    const cover = button("加载预览…", "image-cover", () =>
      showDetail(record.image_id),
    );
    cover.setAttribute("aria-label", `查看图片 #${record.image_id}`);
    covers.push([record, cover]);
    const meta = el("div", "card-meta"),
      title = el("div", "card-title");
    title.append(
      el("strong", "", `#${record.image_id}`),
      el("span", "badge", record.size),
    );
    const name = record.requester_name || record.requester_qq;
    meta.append(
      title,
      el("p", "card-prompt", record.prompt),
      el("div", "card-person", `${name} · ${record.requester_qq}`),
      el("div", "card-date", date(record.created_at)),
    );
    const bottom = el("div", "card-bottom"),
      label = el("label", "inline"),
      check = el("input");
    check.type = "checkbox";
    check.dataset.imageId = record.image_id;
    check.setAttribute("aria-label", `选择 #${record.image_id}`);
    check.addEventListener("change", () => {
      if (check.checked) state.selected.add(record.image_id);
      else state.selected.delete(record.image_id);
      selection();
    });
    label.append(check, document.createTextNode("选择"));
    bottom.append(
      label,
      button("详情 / 编辑", "secondary", () => showDetail(record.image_id)),
    );
    meta.append(bottom);
    card.append(cover, meta);
    $("image-list").append(card);
  }
  pagination($("image-pagination"), data, loadHistory);
  selection();
  // Keep bridge traffic and image decoding bounded, including on large archives.
  let index = 0;
  await Promise.all(
    Array.from({ length: 3 }, async () => {
      while (index < covers.length && epoch === state.epoch) {
        const [record, cover] = covers[index++];
        await preview(record, cover);
      }
    }),
  );
}
async function loadUsers(page = 1) {
  state.userPage = page;
  const epoch = ++state.epoch;
  empty($("user-list"), "正在汇总用户…");
  $("user-pagination").replaceChildren();
  const data = await api("users", {
    search: $("user-filters").elements.search.value.trim(),
    page,
    limit: 24,
  });
  if (epoch !== state.epoch) return;
  $("user-list").replaceChildren();
  if (!data.items.length) empty($("user-list"), "没有匹配的生图用户。");
  for (const user of data.items) {
    const name = user.requester_name || user.requester_qq,
      card = el("article", "user-card");
    card.append(
      el("div", "avatar", [...name][0]),
      el("h2", "", name),
      el("div", "muted", `QQ ${user.requester_qq}`),
      el(
        "p",
        "",
        `${user.count} 张图片 · ${user.group_count} 个群聊\n最近生图：${date(user.latest_at)}`,
      ),
      button("查看与管理图片 →", "secondary", () => {
        $("filters").reset();
        $("filters").elements.requester_qq.value = user.requester_qq;
        state.page = 1;
        location.hash = "history";
      }),
    );
    $("user-list").append(card);
  }
  pagination($("user-pagination"), data, loadUsers);
}
let confirmationResolve = null;
function confirmAction(title, message) {
  $("confirm-title").textContent = title;
  $("confirm-message").textContent = message;
  $("confirm-action").textContent = title;
  $("confirm-dialog").showModal();
  return new Promise((resolve) => {
    confirmationResolve = resolve;
  });
}
function finishConfirmation(value) {
  $("confirm-dialog").close();
  confirmationResolve?.(value);
  confirmationResolve = null;
}
$("cancel-action").onclick = () => finishConfirmation(false);
$("confirm-action").onclick = () => finishConfirmation(true);
$("confirm-dialog").addEventListener("cancel", (event) => {
  event.preventDefault();
  finishConfirmation(false);
});
async function changeDeleted(records, restore = false) {
  const ids = records.map((item) => `#${item.image_id}`).join("、");
  const message = restore
    ? `恢复 ${ids}？图片恢复原编号，不会自动替换任何用户当前的本体参考图。`
    : `将 ${ids} 移入回收站？\n图片会从聊天查询、随机图库和个人画廊中隐藏，关联的本体参考图将解除。原图文件保留，可在回收站恢复。此操作不返还生图配额。`;
  if (!(await confirmAction(restore ? "恢复图片" : "移入回收站", message)))
    return;
  await post(restore ? "images/restore" : "images/delete", {
    items: records.map(({ image_id, revision }) => ({ image_id, revision })),
  });
  cache.clear();
  $("detail-dialog").close();
  await loadHistory(state.page);
  notice(
    restore ? "已恢复所选图片。" : "已移入回收站，可在记录范围中切换查看。",
  );
}
$("batch-action").onclick = () =>
  run(() =>
    changeDeleted(
      state.items.filter((record) => state.selected.has(record.image_id)),
      query().deleted === "1",
    ),
  );
$("select-all").onchange = () => {
  const checked = $("select-all").checked;
  state.selected = new Set(
    checked ? state.items.map((item) => item.image_id) : [],
  );
  document.querySelectorAll("[data-image-id]").forEach((input) => {
    input.checked = checked;
  });
  selection();
};
$("close-detail").onclick = () => $("detail-dialog").close();
async function showDetail(id) {
  const record = await api("detail", { image_id: id });
  $("detail-title").textContent =
    `图库 #${record.image_id}${record.deleted_at ? " · 回收站" : ""}`;
  const grid = el("div", "detail-grid"),
    picture = el("div", "detail-preview", "正在读取预览…"),
    form = el("form", "detail-form"),
    info = el("dl", "detail-info");
  for (const [key, value] of [
    ["用户 QQ", record.requester_qq],
    ["来源群", record.group_id || "私聊"],
    ["生成时间", date(record.created_at)],
    ["生成参数", `${record.size} / ${record.quality}`],
    ["用时", `${record.duration_seconds.toFixed(1)} 秒`],
    ["原始任务", record.job_id ? `#${record.job_id}` : "—"],
  ]) {
    info.append(el("dt", "", key), el("dd", "", value));
  }
  form.append(info);
  const fields = {};
  for (const [key, text, max] of [
    ["requester_name", "显示昵称", 80],
    ["prompt", "图库提示词", 8000],
    ["admin_note", "管理备注（仅管理页可见）", 2000],
  ]) {
    const label = el("label", "", text),
      input = el(key === "requester_name" ? "input" : "textarea");
    input.value = record[key] || "";
    input.maxLength = max;
    input.disabled = !!record.deleted_at;
    if (key === "prompt") {
      input.rows = 6;
      input.required = true;
    }
    label.append(input);
    fields[key] = input;
    form.append(label);
  }
  form.append(
    el(
      "small",
      "muted",
      "编辑只修改图库文字，不重新生图，也不改写原始任务记录。",
    ),
  );
  const actions = el("div", "dialog-actions");
  actions.append(
    button("下载原图", "secondary", () =>
      bridge.download("admin/download", { image_id: record.image_id }),
    ),
  );
  actions.append(
    button(record.deleted_at ? "恢复" : "删除", "danger", () =>
      changeDeleted([record], !!record.deleted_at),
    ),
  );
  if (!record.deleted_at) {
    const save = el("button", "", "保存修改");
    save.type = "submit";
    actions.append(save);
  }
  form.append(actions);
  form.onsubmit = (event) => {
    event.preventDefault();
    run(async () => {
      await post("images/update", {
        image_id: record.image_id,
        revision: record.revision,
        fields: Object.fromEntries(
          Object.entries(fields).map(([key, input]) => [key, input.value]),
        ),
      });
      $("detail-dialog").close();
      cache.clear();
      await loadHistory(state.page);
      notice(`已更新 #${record.image_id}。`);
    }, event.submitter);
  };
  grid.append(picture, form);
  $("detail-body").replaceChildren(grid);
  if (!$("detail-dialog").open) $("detail-dialog").showModal();
  await preview(record, picture, true);
}
const secretPaths = new Set(["api.api_key", "safety.review_api_key"]);
let readers = [];
function field(spec, value, path) {
  const wrap = el(
      "div",
      `field${["text", "template_list"].includes(spec.type) ? " wide" : ""}`,
    ),
    label = el("label", "", spec.description || path);
  wrap.append(label);
  let read;
  if (spec.type === "template_list") {
    const list = el("div", "character-list"),
      entries = [];
    function add(entry = {}) {
      const key = entry.__template_key || "character",
        template = spec.templates[key] || spec.templates.character;
      const card = el("div", "character"),
        tools = el("div", "character-tools"),
        controls = [],
        fields = el("div", "character-fields");
      tools.append(
        el("strong", "", template.name || "人物"),
        button("移除人物", "danger", () => {
          entries.splice(
            entries.findIndex((item) => item.card === card),
            1,
          );
          card.remove();
          markDirty();
        }),
      );
      for (const [name, definition] of Object.entries(template.items)) {
        const control = field(
          definition,
          entry[name] ?? definition.default,
          `${path}.${name}`,
        );
        fields.append(control.node);
        controls.push([name, control]);
      }
      card.append(tools, fields);
      list.append(card);
      entries.push({
        card,
        read: () => ({
          __template_key: key,
          ...Object.fromEntries(
            controls.map(([name, control]) => [name, control.read()]),
          ),
        }),
      });
    }
    if (Array.isArray(value))
      for (const entry of value) {
        if (entry && typeof entry === "object") add(entry);
        else {
          // Preserve older string entries when adding or editing another character.
          const card = el("div", "character"),
            label = el("label", "", "旧版人物条目（名称|别名=图片路径）"),
            input = el("input");
          input.value = String(entry ?? "");
          label.append(input);
          card.append(
            label,
            button("移除人物", "danger", () => {
              entries.splice(
                entries.findIndex((item) => item.card === card),
                1,
              );
              card.remove();
              markDirty();
            }),
          );
          entries.push({ card, read: () => input.value });
          list.append(card);
        }
      }
    const tools = el("div", "character-tools"),
      select = el("select");
    for (const [key, template] of Object.entries(spec.templates)) {
      const option = el("option", "", template.name || key);
      option.value = key;
      select.append(option);
    }
    select.setAttribute("aria-label", "新增人物模板");
    tools.append(
      select,
      button("添加人物", "secondary", () => {
        add({ __template_key: select.value });
        markDirty();
      }),
    );
    wrap.append(list, tools);
    read = () => entries.map((entry) => entry.read());
  } else {
    let input;
    if (spec.options) {
      input = el("select");
      spec.options.forEach((option, i) => {
        const node = el("option", "", spec.labels?.[i] || option);
        node.value = option;
        input.append(node);
      });
      input.value = value ?? spec.default;
      read = () => input.value;
    } else if (spec.type === "bool") {
      input = el("input");
      input.type = "checkbox";
      input.checked = !!value;
      label.className = "inline";
      read = () => input.checked;
    } else if (["list", "file", "text"].includes(spec.type)) {
      input = el("textarea");
      input.rows = 3;
      input.value = Array.isArray(value) ? value.join("\n") : value || "";
      read = () =>
        spec.type === "text"
          ? input.value
          : input.value
              .split(/\r?\n/)
              .map((line) => line.trim())
              .filter(Boolean);
    } else {
      input = el("input");
      input.type = spec.type === "int" ? "number" : "text";
      if (spec.type === "int") input.step = "1";
      input.value = value ?? "";
      read = () =>
        spec.type === "int"
          ? input.value.trim()
            ? Number(input.value)
            : null
          : input.value;
    }
    label.append(input);
    if (secretPaths.has(path)) {
      input.type = "password";
      input.autocomplete = "new-password";
      input.value = "";
      input.disabled = true;
      input.placeholder = state.settings.secrets_set[path]
        ? "已设置（不回显）"
        : "未设置";
      const changeLabel = el(
          "label",
          "inline",
          "替换密钥（勾选后留空表示清空）",
        ),
        change = el("input");
      change.type = "checkbox";
      change.onchange = () => {
        input.disabled = !change.checked;
        markDirty();
      };
      changeLabel.prepend(change);
      wrap.append(changeLabel);
      read = () => (change.checked ? input.value : undefined);
    }
    if (spec.type === "file") {
      const file = el("input");
      file.type = "file";
      file.accept = "image/png,image/jpeg,image/webp";
      file.setAttribute("aria-label", "上传人物参考图");
      file.onchange = () =>
        run(async () => {
          if (!file.files[0]) return;
          const result = await bridge.upload(
            "admin/references/upload",
            file.files[0],
          );
          input.value = [...read(), result.path].join("\n");
          file.value = "";
          markDirty();
          notice("参考图已上传，保存设置后生效。");
        }, file);
      wrap.append(
        file,
        el("small", "", "每行一条现有参考图路径。上传后请保存设置。"),
      );
    } else if (spec.type === "list") wrap.append(el("small", "", "每行一项。"));
  }
  if (spec.hint) wrap.append(el("small", "", spec.hint));
  return { node: wrap, read };
}
function markDirty() {
  state.dirty = true;
  $("settings-state").textContent = "有未保存的修改";
}
$("settings-form").addEventListener("input", markDirty);
$("settings-form").addEventListener("change", markDirty);
function renderSettings(data) {
  state.settings = data;
  state.dirty = false;
  readers = [];
  $("settings-fields").replaceChildren();
  for (const [group, section] of Object.entries(data.schema)) {
    const details = el("details", "panel settings-group");
    details.open = ["api", "generation"].includes(group);
    details.append(el("summary", "", section.description || group));
    const grid = el("div", "settings-grid");
    for (const [key, spec] of Object.entries(section.items)) {
      const control = field(spec, data.values[group][key], `${group}.${key}`);
      readers.push({
        group,
        key,
        control,
        initial: JSON.stringify(control.read()),
      });
      grid.append(control.node);
    }
    details.append(grid);
    $("settings-fields").append(details);
  }
  $("settings-state").textContent = data.busy
    ? "有生图任务运行，完成后可保存"
    : "配置已读取";
}
async function loadSettings() {
  renderSettings(await api("settings"));
}
$("settings-form").onsubmit = (event) => {
  event.preventDefault();
  run(async () => {
    const changes = {};
    for (const { group, key, control, initial } of readers) {
      const value = control.read();
      if (value !== undefined && JSON.stringify(value) !== initial) {
        (changes[group] ||= {})[key] = value;
      }
    }
    if (!Object.keys(changes).length) {
      notice("没有需要保存的修改。");
      return;
    }
    const result = await post("settings/save", {
      revision: state.settings.revision,
      changes,
    });
    renderSettings(result);
    notice("插件设置已保存并生效。");
  }, event.submitter);
};
async function navigate() {
  const requested = location.hash.slice(1) || "history";
  const tab = ["history", "users", "settings"].includes(requested)
    ? requested
    : "history";
  state.tab = tab;
  ++state.epoch;
  notice("");
  document.querySelectorAll("[data-tab]").forEach((link) => {
    link.classList.toggle("active", link.dataset.tab === tab);
    if (link.dataset.tab === tab) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  });
  for (const name of ["history", "users", "settings"])
    $(`${name}-view`).hidden = name !== tab;
  $("stats").hidden = tab !== "history";
  $("page-title").textContent = {
    history: "全部历史",
    users: "按用户管理",
    settings: "插件设置",
  }[tab];
  $("page-description").textContent = {
    history: "按图库编号查看、查询和管理每一张图片。",
    users: "从用户出发，查找属于他们的每一次创作。",
    settings: "在同一个工作台配置生图、审核、人物与使用权限。",
  }[tab];
  if (tab === "history") await loadHistory(state.page);
  else if (tab === "users") await loadUsers(state.userPage);
  else if (!state.settings) await loadSettings();
}
$("filters").onsubmit = (event) => {
  event.preventDefault();
  run(() => loadHistory(1));
};
$("filters").onreset = () => {
  queueMicrotask(() => run(() => loadHistory(1)));
};
$("user-filters").onsubmit = (event) => {
  event.preventDefault();
  run(() => loadUsers(1));
};
$("refresh").onclick = () =>
  run(async () => {
    if (state.tab === "settings") {
      if (
        state.dirty &&
        !(await confirmAction(
          "重新读取配置",
          "放弃本页未保存的修改，重新读取当前配置？",
        ))
      )
        return;
      await loadSettings();
    } else if (state.tab === "users") await loadUsers(state.userPage);
    else {
      cache.clear();
      await loadHistory(state.page);
    }
  }, $("refresh"));
window.addEventListener("hashchange", () => run(navigate));
window.addEventListener("beforeunload", (event) => {
  if (state.dirty) {
    event.preventDefault();
    event.returnValue = "";
  }
});
await run(async () => {
  if (!bridge) {
    $("connection").textContent = "等待 AstrBot WebUI";
    throw new Error(
      "请在 AstrBot 管理面板 → 插件 → GPT Image 2 安全生图 → 可视化视图中打开此页；若无入口，请升级 AstrBot 及配套 WebUI。",
    );
  }
  const context = await bridge.ready();
  document.body.classList.toggle("dark", !!context.isDark);
  if (bridge.onContext)
    bridge.onContext((next) =>
      document.body.classList.toggle("dark", !!next.isDark),
    );
  $("connection").textContent = "● 已连接 AstrBot";
  await navigate();
});
