const state = { token: localStorage.getItem("smartcart_token"), customerId: Number(localStorage.getItem("smartcart_customer")) || null, name: localStorage.getItem("smartcart_name") || "", role: localStorage.getItem("smartcart_role") || "", products: [], cart: [], categories: [], category: "", query: "", authMode: "login" };
const $ = (selector) => document.querySelector(selector);
const money = (value) => new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" }).format(value || 0);
const categoryImages = {
  Electronics: "https://images.unsplash.com/photo-1505740420928-5e560c06d30e?auto=format&fit=crop&w=720&q=82",
  Home: "https://images.unsplash.com/photo-1600210492486-724fe5c67fb0?auto=format&fit=crop&w=720&q=82",
  Books: "https://images.unsplash.com/photo-1544947950-fa07a98d237f?auto=format&fit=crop&w=720&q=82",
  Beauty: "https://images.unsplash.com/photo-1608248543803-ba4f8c70ae0b?auto=format&fit=crop&w=720&q=82",
  Sports: "https://images.unsplash.com/photo-1538805060514-97d9cc17730c?auto=format&fit=crop&w=720&q=82"
};
const imageFor = (product) => categoryImages[product.category] || categoryImages.Home;

async function api(path, options = {}) {
  const response = await fetch(path, { ...options, headers: { ...(options.body ? { "Content-Type": "application/json" } : {}), ...(state.token ? { Authorization: `Bearer ${state.token}` } : {}), ...options.headers } });
  const data = response.status === 204 ? null : await response.json();
  if (!response.ok) throw new Error(data?.detail || "Something went wrong");
  return data;
}

function toast(message) {
  const node = $("#toast"); node.textContent = message; node.classList.add("visible");
  window.clearTimeout(toast.timer); toast.timer = window.setTimeout(() => node.classList.remove("visible"), 2500);
}

async function loadProducts() {
  const params = new URLSearchParams();
  if (state.category) params.set("category", state.category);
  if (state.query) params.set("search", state.query);
  state.products = await api(`/products?${params}`);
  const availableCategories = [...new Set(state.products.map((product) => product.category))];
  if (!state.categories.length) {
    state.categories = [...new Set((await api("/products?limit=200")).map((product) => product.category))];
    $("#category-tabs").innerHTML = `<button class="category-tab active" data-category="" role="tab" aria-selected="true">Everything</button>${state.categories.map((category) => `<button class="category-tab" data-category="${escapeHtml(category)}" role="tab" aria-selected="false">${escapeHtml(category)}</button>`).join("")}`;
  }
  $("#product-count").textContent = `${state.products.length} ${state.products.length === 1 ? "find" : "finds"}`;
  renderProducts();
}

function escapeHtml(value) { return String(value).replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[char]); }

function renderProducts() {
  $("#product-grid").innerHTML = state.products.map((product, index) => `
    <article class="product-card" style="animation-delay:${Math.min(index * 35, 280)}ms">
      <div class="product-image"><img src="${imageFor(product)}" alt="${escapeHtml(product.category)} find" loading="lazy"><span class="category-mark">${escapeHtml(product.category)}</span>${product.discount ? `<span class="discount-mark">-${product.discount}%</span>` : ""}</div>
      <div class="product-info"><h3 class="product-name">${escapeHtml(product.product_name)}</h3><span class="product-price">${money(product.price * (1 - product.discount / 100))}</span><span class="product-meta">${product.stock} in stock</span><span class="product-rating">★ ${Number(product.rating).toFixed(1)}</span><button class="add-button" data-add="${product.product_id}" ${product.stock < 1 ? "disabled" : ""}>${product.stock ? "Add to bag  +" : "Sold out"}</button></div>
    </article>`).join("") || `<p class="empty-state">No finds in this corner yet. Try another search.</p>`;
}

async function loadCart() {
  if (!state.token || !state.customerId) { state.cart = []; renderCart(); return; }
  try { state.cart = await api(`/cart/${state.customerId}`); renderCart(); }
  catch (error) { if (error.message.includes("token")) signOut(); else toast(error.message); }
}

function renderCart() {
  const count = state.cart.reduce((sum, item) => sum + item.quantity, 0);
  const total = state.cart.reduce((sum, item) => sum + item.quantity * item.price * (1 - (item.discount || 0) / 100), 0);
  $("#cart-count").textContent = count; $("#bag-items").textContent = `(${count})`; $("#cart-total").textContent = money(total);
  $("#checkout").disabled = !count || !state.token;
  $("#cart-list").innerHTML = state.cart.length ? state.cart.map((item) => `<div class="cart-item"><div><strong>${escapeHtml(item.product_name)}</strong><br><span>${item.quantity} × ${money(item.price)}</span></div><strong>${money(item.quantity * item.price * (1 - (item.discount || 0) / 100))}<button aria-label="Remove ${escapeHtml(item.product_name)}" data-remove="${item.cart_id}">×</button></strong></div>`).join("") : `<p class="empty-state">${state.token ? "Your next great find goes here." : "Sign in to start your bag."}</p>`;
}

async function loadRecommendations() {
  const list = $("#recommendation-list");
  if (!state.token || !state.customerId) { $("#recommendation-note").textContent = "Sign in to see your personal edit."; list.innerHTML = ""; return; }
  try {
    const data = await api(`/recommendations/${state.customerId}`);
    $("#recommendation-note").textContent = "A few good things, based on what you love.";
    list.innerHTML = data.recommendations.map((item) => {
      const product = state.products.find((candidate) => candidate.product_id === item.product_id) || { category: "Home" };
      return `<div class="recommendation-item"><div class="recommendation-thumb"><img src="${imageFor(product)}" alt="" loading="lazy"></div><div><strong>${escapeHtml(item.product_name)}</strong><small>${Math.round(item.score * 100)}% match</small></div><button class="recommendation-add" aria-label="Add ${escapeHtml(item.product_name)}" data-add="${item.product_id}">+</button></div>`;
    }).join("");
  } catch (error) { $("#recommendation-note").textContent = error.message; }
}

async function loadDashboard() {
  if (!state.token || !state.customerId) { $("#dashboard").classList.add("hidden"); return; }
  try {
    const data = await api(`/customer-dashboard/${state.customerId}`);
    const likelyProducts = data.purchase_predictions
      .map((prediction) => ({ ...prediction, product: data.recommended_products.find((item) => item.product_id === prediction.product_id) }))
      .filter((prediction) => prediction.product)
      .sort((left, right) => right.purchase_probability - left.purchase_probability)
      .slice(0, 3);
    const predictionMarkup = likelyProducts.map((prediction) => `<div class="prediction-row"><span>${escapeHtml(prediction.product.product_name)}</span><strong>${Math.round(prediction.purchase_probability * 100)}%</strong></div>`).join("");
    $("#dashboard-content").innerHTML = `<span class="segment-tag">${escapeHtml(data.customer_segment)}</span><div class="dashboard-stats"><div class="dashboard-stat"><span>Orders</span><strong>${data.total_orders}</strong></div><div class="dashboard-stat"><span>Spent</span><strong>${money(data.total_spending)}</strong></div><div class="dashboard-stat"><span>Average order</span><strong>${money(data.average_order_value)}</strong></div><div class="dashboard-stat"><span>Purchase signals</span><strong>${data.purchase_predictions.length}</strong></div></div><div class="prediction-list"><span class="rail-kicker">MOST LIKELY NEXT</span>${predictionMarkup || `<p class="empty-state">More browsing will sharpen your picks.</p>`}</div>`;
    $("#dashboard").classList.remove("hidden");
  } catch (error) { toast(error.message); }
}

function setAuth(mode) {
  state.authMode = mode;
  $("#auth-title").textContent = mode === "login" ? "Sign in" : "Create your account";
  $("#auth-submit").textContent = mode === "login" ? "Sign in" : "Create account";
  $("#name-field").classList.toggle("hidden", mode === "login");
  $("#age-field").classList.toggle("hidden", mode === "login");
  $("#city-field").classList.toggle("hidden", mode === "login");
  $("#auth-form [name=password]").autocomplete = mode === "login" ? "current-password" : "new-password";
  document.querySelectorAll(".auth-mode").forEach((button) => button.classList.toggle("active", button.dataset.mode === mode));
  $("#auth-message").textContent = "";
}

async function onAuth(event) {
  event.preventDefault();
  const values = Object.fromEntries(new FormData(event.currentTarget));
  const payload = state.authMode === "login" ? { email: values.email, password: values.password } : { ...values, age: Number(values.age) };
  try {
    if (state.authMode === "register") {
      const registered = await api("/register", { method: "POST", body: JSON.stringify(payload) });
      state.token = registered.access_token; state.customerId = registered.customer.customer_id; state.name = registered.customer.full_name; state.role = "customer";
    } else {
      const loggedIn = await api("/login", { method: "POST", body: JSON.stringify(payload) });
      state.token = loggedIn.access_token; state.customerId = loggedIn.customer_id; state.role = loggedIn.role; state.name = values.email.split("@")[0];
    }
    localStorage.setItem("smartcart_token", state.token); localStorage.setItem("smartcart_customer", state.customerId || ""); localStorage.setItem("smartcart_name", state.name); localStorage.setItem("smartcart_role", state.role);
    $("#auth-dialog").close(); updateAccount(); await loadCart(); await loadRecommendations(); await loadDashboard(); toast("You're in. Your picks are ready.");
  } catch (error) { $("#auth-message").textContent = error.message; }
}

function updateAccount() {
  $("#account-label").textContent = state.token ? state.name || "Welcome back" : "Guest";
  $("#account-button").textContent = state.token ? "Account" : "Sign in";
}

function signOut() {
  state.token = null; state.customerId = null; state.name = ""; state.role = "";
  ["smartcart_token", "smartcart_customer", "smartcart_name", "smartcart_role"].forEach((key) => localStorage.removeItem(key));
  $("#dashboard").classList.add("hidden"); updateAccount(); renderCart(); loadRecommendations(); toast("Signed out.");
}

document.addEventListener("click", async (event) => {
  const categoryButton = event.target.closest("[data-category]");
  if (categoryButton) {
    state.category = categoryButton.dataset.category;
    document.querySelectorAll(".category-tab").forEach((button) => { button.classList.toggle("active", button === categoryButton); button.setAttribute("aria-selected", button === categoryButton ? "true" : "false"); });
    await loadProducts(); return;
  }
  const addButton = event.target.closest("[data-add]");
  if (addButton) {
    if (!state.token) { $("#auth-dialog").showModal(); return; }
    try { await api("/cart", { method: "POST", body: JSON.stringify({ customer_id: state.customerId, product_id: Number(addButton.dataset.add), quantity: 1 }) }); await loadCart(); toast("Added to your bag."); }
    catch (error) { toast(error.message); }
  }
  const removeButton = event.target.closest("[data-remove]");
  if (removeButton) { try { await api(`/cart/${removeButton.dataset.remove}`, { method: "DELETE" }); await loadCart(); } catch (error) { toast(error.message); } }
});

$("#account-button").addEventListener("click", () => {
  if (!state.token) { setAuth("login"); $("#auth-dialog").showModal(); }
  else { $("#dashboard").classList.toggle("hidden"); $("#dashboard").scrollIntoView({ behavior: "smooth", block: "center" }); }
});
$("#auth-form").addEventListener("submit", onAuth);
document.querySelectorAll(".auth-mode").forEach((button) => button.addEventListener("click", () => setAuth(button.dataset.mode)));
$("#search").addEventListener("input", (event) => { window.clearTimeout(loadProducts.timer); state.query = event.target.value; loadProducts.timer = window.setTimeout(() => loadProducts().catch((error) => toast(error.message)), 180); });
$("#cart-jump").addEventListener("click", () => $("#cart").scrollIntoView({ behavior: "smooth", block: "center" }));
$("#signout").addEventListener("click", signOut);
$("#checkout").addEventListener("click", async () => {
  const message = $("#checkout-message");
  try {
    const order = await api("/orders", { method: "POST", body: JSON.stringify({ payment_method: $("#payment-method").value }) });
    message.textContent = `Order #${order.order_id} placed. A good choice.`; await loadCart(); await loadProducts(); await loadRecommendations(); await loadDashboard(); toast("Order placed. Thanks for shopping thoughtfully.");
  } catch (error) { message.textContent = error.message; }
});
document.addEventListener("keydown", (event) => { if (event.key === "/" && !["INPUT", "TEXTAREA"].includes(document.activeElement.tagName)) { event.preventDefault(); $("#search").focus(); } });

updateAccount();
Promise.all([loadProducts(), loadCart(), loadRecommendations()]).catch((error) => toast(error.message));
if (state.token) loadDashboard();