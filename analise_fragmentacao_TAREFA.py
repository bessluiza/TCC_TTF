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
