import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import kruskal, spearmanr, entropy, t as tdist
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

OUT = "."
FIG = OUT + "/figuras"

SOC_NAMES = {
 "11":"Gestão","13":"Negócios/Financeiro","15":"Computação/Matemática","17":"Arquitetura/Engenharia",
 "19":"Ciências","21":"Serviço social","23":"Jurídico","25":"Educação","27":"Artes/Mídia",
 "29":"Saúde (técnicos)","31":"Apoio à saúde","33":"Segurança","35":"Alimentação","37":"Limpeza/Manut.",
 "39":"Cuidados pessoais","41":"Vendas","43":"Administrativo","45":"Agro","47":"Construção",
 "49":"Manutenção/Reparo","51":"Produção","53":"Transporte","55":"Militar"}
KW = {"11","13","15","23","43"}   # recorte "trabalho intensivo em informação"

# ---------- carregar ----------
occ = pd.read_csv("occ_level.csv")
occ["soc2"] = occ["O*NET-SOC Code"].str[:2]
occ["soc6"] = occ["O*NET-SOC Code"].str.split(".").str[0]
occ["knowledge_work"] = occ["soc2"].isin(KW)

# proxy de TTF
occ["exposicao"] = occ["dv_rating_beta"]

# ---------- (3) FRAGMENTAÇÃO via entropia das categorias E0/E1/E2 ----------
a = occ["dv_rating_alpha"].clip(0,1)              # E1
g = occ["dv_rating_gamma"].clip(0,1)              # E1+E2
g = np.maximum(g, a)                              # garante gamma>=alpha
E1 = a; E2 = (g - a).clip(lower=0); E0 = (1 - g).clip(lower=0)
P = np.vstack([E0, E1, E2]).T
P = P / P.sum(axis=1, keepdims=True)
occ["fragmentacao"] = [entropy(p) / np.log(3) for p in P]   # 0..1 (1 = mais "serrilhado")

# ---------- (2) descritiva por grande grupo ----------
desc = (occ.groupby("soc2")["exposicao"].agg(["count","mean"])
           .assign(grupo=lambda d: d.index.map(SOC_NAMES))
           .sort_values("mean", ascending=False))
desc.to_csv(FIG+"/tab_exposicao_por_grupo.csv")

plt.figure(figsize=(9,7))
d2 = desc[desc["count"]>=5]
cores = ["#534AB7" if s in KW else "#B4B2A9" for s in d2.index]
plt.barh(d2["grupo"], d2["mean"], color=cores)
plt.xlabel("Exposição média a LLM (proxy de TTF, medida beta)")
plt.title("Ajuste médio por grande grupo ocupacional\n(roxo = trabalho intensivo em informação)")
plt.gca().invert_yaxis(); plt.tight_layout()
plt.savefig(FIG+"/fig_exposicao_por_grupo.png", dpi=200); plt.close()

# teste KW vs outras (exposição e fragmentação)
for var in ["exposicao","fragmentacao"]:
    kw = occ.loc[occ.knowledge_work, var].dropna()
    ot = occ.loc[~occ.knowledge_work, var].dropna()
    H,p = kruskal(kw, ot)
    print(f"[{var}] mediana KW={kw.median():.3f} | outras={ot.median():.3f} | Kruskal-Wallis H={H:.1f} p={p:.2e}")

# figura Fronteira Fragmentada: exposição vs fragmentação
plt.figure(figsize=(8.5,6.5))
o = occ
plt.scatter(o.loc[~o.knowledge_work,"exposicao"], o.loc[~o.knowledge_work,"fragmentacao"],
            s=16, alpha=.35, color="#B4B2A9", label="Outras ocupações")
plt.scatter(o.loc[o.knowledge_work,"exposicao"], o.loc[o.knowledge_work,"fragmentacao"],
            s=22, alpha=.75, color="#534AB7", label="Trabalho intensivo em informação")
plt.xlabel("Exposição média da ocupação (proxy de TTF)")
plt.ylabel("Fragmentação intra-ocupação (entropia E0/E1/E2)")
plt.title("Fronteira Tecnológica Fragmentada\no ajuste se distribui de forma desigual entre as tarefas")
plt.legend(); plt.tight_layout()
plt.savefig(FIG+"/fig_fronteira_fragmentada.png", dpi=200); plt.close()

# ---------- (4) REGRESSÃO OLS (numpy) : exposição ~ atividades ----------
wa = pd.read_excel("Work_Activities.xlsx", sheet_name="Work Activities")
wa = wa[wa["Scale ID"]=="IM"]
wide = wa.pivot_table(index="O*NET-SOC Code", columns="Element Name", values="Data Value", aggfunc="mean")
preditores = {
 "Getting Information":"Obter informação",
 "Analyzing Data or Information":"Analisar dados",
 "Processing Information":"Processar informação",
 "Documenting/Recording Information":"Documentar informação",
 "Working with Computers":"Trabalhar com computadores",
 "Performing General Physical Activities":"Atividade física geral",
 "Handling and Moving Objects":"Manusear objetos",
}
cols = [c for c in preditores if c in wide.columns]
base = wide[cols].join(occ.set_index("O*NET-SOC Code")["exposicao"], how="inner").dropna()
Xraw = base[cols].values
y = base["exposicao"].values
# padroniza preditores (coeficientes comparáveis)
Xz = StandardScaler().fit_transform(Xraw)
X = np.column_stack([np.ones(len(Xz)), Xz])
bhat, *_ = np.linalg.lstsq(X, y, rcond=None)
resid = y - X@bhat
n,k = X.shape
sigma2 = resid@resid/(n-k)
XtX_inv = np.linalg.inv(X.T@X)
se = np.sqrt(np.diag(sigma2*XtX_inv))
tval = bhat/se
pval = 2*(1-tdist.cdf(np.abs(tval), df=n-k))
ss_tot = ((y-y.mean())**2).sum(); R2 = 1 - (resid@resid)/ss_tot
nomes = ["(intercepto)"]+[preditores[c] for c in cols]
reg = pd.DataFrame({"variavel":nomes,"coef_padronizado":bhat,"erro_padrao":se,"t":tval,"p_valor":pval})
reg.to_csv(FIG+"/tab_regressao_ols.csv", index=False)
print(f"\n[Regressão OLS] n={n}  R²={R2:.3f}")
print(reg.round(3).to_string(index=False))

# figura coeficientes
plt.figure(figsize=(8.5,5.5))
rr = reg[reg.variavel!="(intercepto)"].sort_values("coef_padronizado")
cor = ["#1D9E75" if c>0 else "#D85A30" for c in rr["coef_padronizado"]]
plt.barh(rr["variavel"], rr["coef_padronizado"], color=cor, xerr=rr["erro_padrao"], capsize=3)
plt.axvline(0, color="#444", lw=.8)
plt.xlabel("Efeito padronizado sobre a exposição (proxy de TTF)")
plt.title(f"O que prediz o ajuste tarefa-tecnologia (R²={R2:.2f})\nverde = aumenta o ajuste | laranja = reduz")
plt.tight_layout(); plt.savefig(FIG+"/fig_regressao_coeficientes.png", dpi=200); plt.close()

# ---------- (5) CLUSTERS ----------
feat = occ[["exposicao","fragmentacao"]].dropna()
Xs = StandardScaler().fit_transform(feat.values)
sil = {kk: silhouette_score(Xs, KMeans(kk, n_init=10, random_state=42).fit_predict(Xs)) for kk in range(2,7)}
kbest = max(sil, key=sil.get)
km = KMeans(kbest, n_init=10, random_state=42).fit(Xs)
feat = feat.copy(); feat["cluster"] = km.labels_
ordem = feat.groupby("cluster")["exposicao"].mean().sort_values().index
rotulos = {c:r for c,r in zip(ordem, ["Baixo ajuste","Ajuste médio","Alto ajuste"][:kbest])}
feat["perfil"] = feat["cluster"].map(rotulos)
print(f"\n[Clusters] k escolhido={kbest} (silhueta={sil[kbest]:.3f})")
print(feat.groupby("perfil")["exposicao"].agg(["count","mean"]).round(3).to_string())
plt.figure(figsize=(8.5,6.5))
for perfil,cor in zip(["Baixo ajuste","Ajuste médio","Alto ajuste"],["#B4B2A9","#EF9F27","#1D9E75"]):
    s = feat[feat.perfil==perfil]
    if len(s): plt.scatter(s["exposicao"], s["fragmentacao"], s=18, alpha=.6, color=cor, label=perfil)
plt.xlabel("Exposição (proxy de TTF)"); plt.ylabel("Fragmentação intra-ocupação")
plt.title(f"Perfis de tarefa por nível de ajuste (k={kbest})"); plt.legend(); plt.tight_layout()
plt.savefig(FIG+"/fig_clusters.png", dpi=200); plt.close()

# ---------- (6) VALIDAÇÃO com AIOE ----------
aioe = pd.read_excel("AIOE_DataAppendix.xlsx", sheet_name="Appendix A")
aioe.columns = ["soc6","titulo","AIOE"]
aioe["AIOE"] = pd.to_numeric(aioe["AIOE"], errors="coerce")
occ6 = occ.groupby("soc6")["exposicao"].mean().reset_index()
m = occ6.merge(aioe[["soc6","AIOE"]], on="soc6", how="inner").dropna()
rho,pv = spearmanr(m["exposicao"], m["AIOE"])
print(f"\n[Validação] Spearman exposição(Eloundou) x AIOE(Felten): rho={rho:.3f} p={pv:.2e} (n={len(m)})")
plt.figure(figsize=(7.5,6))
plt.scatter(m["exposicao"], m["AIOE"], s=16, alpha=.4, color="#185FA5")
plt.xlabel("Exposição a LLM (Eloundou et al., 2024)"); plt.ylabel("AIOE (Felten et al., 2021)")
plt.title(f"Validação convergente do proxy de TTF\nSpearman ρ = {rho:.2f} (n={len(m)})")
plt.tight_layout(); plt.savefig(FIG+"/fig_validacao_aioe.png", dpi=200); plt.close()

occ.to_csv(FIG+"/base_ocupacoes_processada.csv", index=False)
print("\nFiguras e tabelas salvas em", FIG)
