function includeHTML() {
  const items = document.querySelectorAll("[data-include]");

  items.forEach(item => {
    const file = item.getAttribute("data-include");

    fetch(file)
      .then(response => response.text())
      .then(content => {
        item.innerHTML = content;
      })
      .catch(() => {
        item.innerHTML = "Error loading include file";
      });
  });
}

document.addEventListener("DOMContentLoaded", includeHTML);