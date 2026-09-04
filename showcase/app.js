document.documentElement.classList.add("js");

const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

const revealItems = document.querySelectorAll(".reveal");
document.querySelectorAll(".hero .reveal").forEach((item) => item.classList.add("in-view"));
if (reducedMotion || !("IntersectionObserver" in window)) {
  revealItems.forEach((item) => item.classList.add("in-view"));
} else {
  const revealObserver = new IntersectionObserver(
    (entries, observer) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        entry.target.classList.add("in-view");
        observer.unobserve(entry.target);
      });
    },
    { threshold: 0.12 },
  );
  revealItems.forEach((item) => revealObserver.observe(item));
}

const chartRows = [...document.querySelectorAll(".bar-row")];
const chartTitle = document.querySelector("#chart-title");
const chartNote = document.querySelector("#chart-note");
const metricButtons = document.querySelectorAll("[data-metric]");

const chartFormats = {
  memory: {
    title: "Peak GPU memory",
    note: "Process VRAM sampled every 100 ms. Lower is better.",
    format: (value) => `${Math.round(value).toLocaleString()} MiB`,
  },
  latency: {
    title: "Compiled batch latency",
    note: "One generated image after a compilation warmup. Lower is better.",
    format: (value) => `${value.toFixed(2)} s`,
  },
};

function renderChart(metric) {
  const values = chartRows.map((row) => Number(row.dataset[metric]));
  const maximum = Math.max(...values);
  chartRows.forEach((row) => {
    const value = Number(row.dataset[metric]);
    const fill = row.querySelector(".bar-fill");
    const output = row.querySelector("output");
    requestAnimationFrame(() => {
      fill.style.width = `${Math.max(8, (value / maximum) * 100)}%`;
    });
    output.textContent = chartFormats[metric].format(value);
  });
  chartTitle.textContent = chartFormats[metric].title;
  chartNote.textContent = chartFormats[metric].note;
}

metricButtons.forEach((button) => {
  button.addEventListener("click", () => {
    metricButtons.forEach((item) => item.classList.toggle("is-active", item === button));
    renderChart(button.dataset.metric);
  });
});
renderChart("memory");

const demoTabs = document.querySelectorAll("[data-demo]");
const demoPanels = document.querySelectorAll(".demo-panel");
demoTabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    const target = `panel-${tab.dataset.demo}`;
    demoTabs.forEach((item) => {
      const selected = item === tab;
      item.classList.toggle("is-active", selected);
      item.setAttribute("aria-selected", String(selected));
    });
    demoPanels.forEach((panel) => {
      const selected = panel.id === target;
      panel.classList.toggle("is-active", selected);
      panel.hidden = !selected;
      if (!selected) panel.querySelector("video")?.pause();
    });
  });
});

const outputButtons = document.querySelectorAll("[data-output]");
const t2iImage = document.querySelector("#t2i-image");
const t2iMode = document.querySelector("#t2i-mode");
const outputLabels = {
  bf16: "BF16",
  int8wo: "INT8WO",
  int8dq: "INT8DQ",
};
outputButtons.forEach((button) => {
  button.addEventListener("click", () => {
    outputButtons.forEach((item) => item.classList.toggle("is-active", item === button));
    const mode = button.dataset.output;
    t2iImage.classList.add("is-switching");
    const swap = () => {
      t2iImage.src = button.dataset.src;
      t2iImage.alt = `${outputLabels[mode]} robotics laboratory generation`;
      t2iMode.textContent = outputLabels[mode];
      t2iImage.classList.remove("is-switching");
    };
    if (reducedMotion) swap();
    else window.setTimeout(swap, 120);
  });
});

const toast = document.querySelector(".toast");
let toastTimer;

function showToast() {
  window.clearTimeout(toastTimer);
  toast.classList.add("is-visible");
  toastTimer = window.setTimeout(() => toast.classList.remove("is-visible"), 1800);
}

async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    const area = document.createElement("textarea");
    area.value = text;
    area.style.position = "fixed";
    area.style.opacity = "0";
    document.body.append(area);
    area.select();
    document.execCommand("copy");
    area.remove();
  }
  showToast();
}

document.querySelectorAll(".copy-button").forEach((button) => {
  button.addEventListener("click", () => {
    const source = button.dataset.copy ? document.getElementById(button.dataset.copy)?.textContent : button.dataset.copyText;
    if (source) copyText(source.trim());
  });
});

t2iImage.addEventListener("load", () => t2iImage.classList.remove("is-switching"));
