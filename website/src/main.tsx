import { StrictMode, useState } from "react";
import { createRoot } from "react-dom/client";
import "./style.css";

const kod = `import totaai as ta

model = ta.Model()
model.dodaj(
    ta.WarstwaLiniowa(784, 128),
    ta.ReLU(),
    ta.WarstwaLiniowa(128, 10),
)
model.skompiluj(ta.MSE(), ta.Adam())
model.trenuj(dane, etykiety, epoki=10)`;

function App() {
  const [skopiowano, ustawSkopiowano] = useState(false);
  const kopiuj = async () => { await navigator.clipboard.writeText("pip install totaai"); ustawSkopiowano(true); window.setTimeout(() => ustawSkopiowano(false), 1600); };
  return <main>
    <nav><a className="logo" href="#top"><span>◆</span> TotaAI</a><div className="links"><a href="#mozliwosci">Możliwości</a><a href="#dokumentacja">Dokumentacja</a><a className="github" href="https://github.com/totatomasz13-web/TotaAI">GitHub ↗</a></div></nav>
    <section className="hero" id="top"><div className="eyebrow">POLSKI FRAMEWORK AI · v0.2.0</div><h1>Buduj AI.<br /><em>Po swojemu.</em></h1><p className="lead">Czytelna biblioteka uczenia maszynowego z polskim API. Od pierwszego neuronu do własnego modelu, bez bariery terminologii.</p><div className="actions"><a className="primary" href="#start">Zacznij budować <span>→</span></a><a className="hero-docs" href="#dokumentacja">Dokumentacja API ↗</a><button className="secondary copy" onClick={kopiuj}>{skopiowano ? "Skopiowano ✓" : "pip install totaai"}</button></div></section>
    <section className="code-card" id="start"><div className="dots"><i></i><i></i><i></i><small>przyklad.py</small></div><pre><code>{kod}</code></pre></section>
    <section className="features" id="mozliwosci"><div className="section-title"><span>01 / DLACZEGO TOTA</span><h2>Mniej składni.<br /><em>Więcej pomysłów.</em></h2></div><div className="grid"><article><b>01</b><h3>Polskie API</h3><p>WarstwaLiniowa, trenuj, przewidz. Nazwy, które mówią co robią.</p></article><article><b>02</b><h3>Autodiff w środku</h3><p>Automatyczne gradienty i gotowe optymalizatory bez ręcznej matematyki.</p></article><article><b>03</b><h3>Gotowe do GPU</h3><p>Ten sam prosty interfejs dla eksperymentów na CPU i akceleratorach.</p></article><article><b>04</b><h3>Otwarty rozwój</h3><p>MIT, typowane moduły i testy. Twórz własne warstwy oraz architektury.</p></article></div></section>
    <section className="docs" id="dokumentacja"><div className="section-title"><span>02 / DOKUMENTACJA API</span><h2>Od importu<br /><em>do treningu.</em></h2></div><div className="docs-content"><div className="doc-block"><code>pip install totaai</code><p>Zainstaluj stabilną wersję biblioteki i zacznij od prostego modelu.</p></div><div className="doc-block"><h3>Model i trening</h3><pre><code>{`model = ta.Model()\nmodel.dodaj(ta.WarstwaLiniowa(2, 8), ta.ReLU())\nmodel.skompiluj(ta.MSE(), ta.Adam())\nmodel.trenuj(dane, etykiety, epoki=10)`}</code></pre></div><div className="doc-block"><h3>Dostępne moduły</h3><div className="api-list"><span>Tensor + autodiff</span><span>WarstwaLiniowa</span><span>ReLU · Sigmoid · Softmax</span><span>MSE · EntropiaKrzyzowa</span><span>SGD · Adam</span><span>Batchowanie i walidacja</span></div></div><a className="docs-link" href="https://github.com/totatomasz13-web/TotaAI/blob/main/docs/API.md">Przeczytaj pełną dokumentację na GitHubie ↗</a></div></section>
    <footer><div className="logo"><span>◆</span> TotaAI</div><p>AI, które mówi po Twojemu.</p><a href="https://github.com/totatomasz13-web/TotaAI">github.com/totatomasz13-web/TotaAI ↗</a></footer>
  </main>;
}

createRoot(document.getElementById("root")!).render(<StrictMode><App /></StrictMode>);
