# =====================================================================
# Apêndice — Código completo da análise quantitativa de dados secundários
# Dados: github.com/openai/GPTs-are-GPTs (Eloundou et al.); O*NET; AIOE (Felten et al.)
# =====================================================================

## PARTE 1 — Análise no nível ocupacional, regressão, clusters e validação
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


## PARTE 2 — Fragmentação no nível de tarefa (Fronteira Tecnológica Fragmentada)
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import kruskal, entropy

FIG = "./figuras"
KW = {"11","13","15","23","43"}
SOC_NAMES = {"11":"Gestão","13":"Negócios/Financeiro","15":"Computação/Matemática","23":"Jurídico","43":"Administrativo"}

df = pd.read_csv("full_labelset.tsv", sep="\t")
df = df[["O*NET-SOC Code","Task ID","Title","gpt4_exposure","beta"]].dropna(subset=["beta"])
df["soc2"] = df["O*NET-SOC Code"].str[:2]
df["knowledge_work"] = df["soc2"].isin(KW)

# ---- estatísticas por ocupação a partir das TAREFAS ----
def occ_stats(g):
    b = g["beta"].values
    cats = g["gpt4_exposure"].value_counts(normalize=True)
    p = np.array([cats.get("E0",0), cats.get("E2",0), cats.get("E1",0)])
    p = p[p>0]
    return pd.Series({
        "n_tarefas": len(b),
        "exposicao_media": b.mean(),
        "desvio": b.std(ddof=0),
        "entropia": entropy(p)/np.log(3) if len(p)>1 else 0.0,
        "share_E0": (g["gpt4_exposure"]=="E0").mean(),   # fração de tarefas SEM ajuste
    })

occ = df.groupby("O*NET-SOC Code").apply(occ_stats, include_groups=False).reset_index()
occ["soc2"] = occ["O*NET-SOC Code"].str[:2]
occ["knowledge_work"] = occ["soc2"].isin(KW)
occ = occ[occ["n_tarefas"]>=5]

# ---- teste: fragmentação (desvio das tarefas) difere entre KW e outras? ----
for var in ["desvio","entropia","exposicao_media"]:
    kw = occ.loc[occ.knowledge_work,var]; ot = occ.loc[~occ.knowledge_work,var]
    H,p = kruskal(kw,ot)
    print(f"[{var}] mediana KW={kw.median():.3f} | outras={ot.median():.3f} | KW H={H:.1f} p={p:.2e}")

# Achado-chave da Fronteira Fragmentada: mesmo em ocupações de ALTO ajuste, há tarefas SEM ajuste
alto = occ[occ.exposicao_media>=0.6]
print(f"\nOcupações de alto ajuste (média>=0.6): {len(alto)}")
print(f"  fração média de tarefas SEM exposição (E0) nessas ocupações: {alto.share_E0.mean():.1%}")
print(f"  ou seja: mesmo nos cargos mais 'encaixados', em média {alto.share_E0.mean():.0%} das tarefas ficam fora da fronteira")

# ---- FIGURA 1: média vs desvio (a fronteira é serrilhada em todo o espectro) ----
plt.figure(figsize=(8.5,6.5))
plt.scatter(occ.loc[~occ.knowledge_work,"exposicao_media"], occ.loc[~occ.knowledge_work,"desvio"],
            s=16, alpha=.35, color="#B4B2A9", label="Outras ocupações")
plt.scatter(occ.loc[occ.knowledge_work,"exposicao_media"], occ.loc[occ.knowledge_work,"desvio"],
            s=24, alpha=.75, color="#534AB7", label="Trabalho intensivo em informação")
plt.xlabel("Exposição média das tarefas da ocupação (proxy de TTF)")
plt.ylabel("Desvio-padrão entre as tarefas (fragmentação)")
plt.title("Fronteira Tecnológica Fragmentada (nível de tarefa)\no ajuste varia entre as tarefas dentro de cada ocupação")
plt.legend(); plt.tight_layout()
plt.savefig(FIG+"/fig_fronteira_fragmentada_TAREFA.png", dpi=200); plt.close()

# ---- FIGURA 2: distribuição de exposição DENTRO de ocupações exemplares ----
exemplos_soc = {
 "15-1252.00":"Desenvolvedores de software","13-2011.00":"Contadores e auditores",
 "23-1011.00":"Advogados","43-3031.00":"Aux. de contabilidade",
 "11-1011.00":"Diretores executivos","29-1141.00":"Enfermeiros"}
rows=[]
for soc,nome in exemplos_soc.items():
    g = df[df["O*NET-SOC Code"]==soc]
    if len(g)==0: continue
    s = g["gpt4_exposure"].value_counts(normalize=True)
    rows.append((nome, s.get("E0",0), s.get("E2",0), s.get("E1",0)))
ex = pd.DataFrame(rows, columns=["ocup","E0","E2","E1"]).set_index("ocup")
ex = ex.sort_values("E1")
plt.figure(figsize=(9,5.5))
left = np.zeros(len(ex))
for col,cor,lab in [("E0","#D85A30","Sem ajuste (E0)"),("E2","#EF9F27","Ajuste com software (E2)"),("E1","#1D9E75","Ajuste direto (E1)")]:
    plt.barh(ex.index, ex[col], left=left, color=cor, label=lab); left+=ex[col].values
plt.xlabel("Fração das tarefas da ocupação")
plt.title("Dentro de um mesmo cargo, o ajuste é 'serrilhado'\ncada barra é uma ocupação; as cores são tarefas de ajuste diferente")
plt.legend(loc="upper center", bbox_to_anchor=(0.5,-0.12), ncol=3, fontsize=9, frameon=False); plt.xlim(0,1); plt.tight_layout()
plt.savefig(FIG+"/fig_jaggedness_dentro_do_cargo.png", dpi=200); plt.close()

occ.to_csv(FIG+"/base_fragmentacao_TAREFA.csv", index=False)
print("\nFiguras salvas.")
