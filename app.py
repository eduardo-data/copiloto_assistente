from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from src.azure_client import AzureLLM
from src.config import Settings
from src.rag import LocalRAG

st.set_page_config(page_title="Copiloto Assistente", layout="wide")
st.title("Copiloto Assistente")
st.caption("MVP de Agent Assist com RAG local e simulação sintética")

settings = Settings()
rag = LocalRAG()


def build_context(results: list[dict]) -> str:
    return "\n\n".join(
        f"FONTE: {item['source']} | SCORE: {item['score']:.3f}\n{item['text']}"
        for item in results
    )


def assist(message: str, history: list[dict] | None = None) -> tuple[str, list[dict]]:
    results = rag.search(message, settings.top_k)
    context = build_context(results)
    history_text = json.dumps(history or [], ensure_ascii=False)
    llm = AzureLLM(settings)
    system = """Você é um copiloto para operadores de atendimento.
Use somente as fontes fornecidas. Não invente políticas, preços, prazos ou procedimentos.
Caso não exista evidência suficiente, informe claramente que o operador deve consultar um supervisor.
Responda em português do Brasil e devolva:
1. Intenção detectada
2. Resposta sugerida ao cliente
3. Próxima ação do operador
4. Alertas
5. Fontes utilizadas
"""
    user = f"HISTÓRICO:\n{history_text}\n\nMENSAGEM ATUAL:\n{message}\n\nFONTES:\n{context or 'Nenhuma fonte recuperada.'}"
    return llm.complete(system, user), results

manual_tab, simulation_tab, docs_tab = st.tabs(
    ["Atendimento manual", "Simulação sintética", "Base documental"]
)

with manual_tab:
    message = st.text_area("Mensagem do cliente", "Quero cancelar meu plano porque está caro.")
    if st.button("Gerar assistência", type="primary"):
        try:
            answer, results = assist(message)
            st.subheader("Sugestão ao operador")
            st.markdown(answer)
            st.subheader("Trechos recuperados")
            for item in results:
                with st.expander(f"{item['source']} — score {item['score']:.3f}"):
                    st.write(item["text"])
        except Exception as exc:
            st.error(f"Falha ao gerar assistência: {exc}")

with simulation_tab:
    st.write("Executa uma conversa curta entre um cliente sintético e um operador apoiado pelo RAG.")
    if st.button("Executar cenário sintético"):
        try:
            scenario_path = Path("data/scenarios/cancelamento.json")
            scenario = json.loads(scenario_path.read_text(encoding="utf-8"))
            llm = AzureLLM(settings)
            conversation: list[dict] = []
            customer_message = scenario["initial_message"]

            for turn in range(min(settings.max_turns, 6)):
                conversation.append({"role": "cliente", "content": customer_message})
                operator_answer, _ = assist(customer_message, conversation)
                conversation.append({"role": "operador", "content": operator_answer})

                if turn >= 2:
                    break

                customer_message = llm.complete(
                    "Você representa um cliente sintético. Continue a conversa de modo realista, sem revelar o roteiro. Responda apenas como cliente.",
                    f"Cenário: {json.dumps(scenario, ensure_ascii=False)}\nConversa: {json.dumps(conversation, ensure_ascii=False)}",
                )

            for item in conversation:
                with st.chat_message("user" if item["role"] == "cliente" else "assistant"):
                    st.markdown(f"**{item['role'].title()}:** {item['content']}")

            evaluation = llm.complete(
                "Você é um juiz de qualidade de atendimento. Avalie fundamentação, aderência ao procedimento, resolução, riscos e alucinações. Dê notas de 0 a 10 e findings objetivos.",
                json.dumps({"scenario": scenario, "conversation": conversation}, ensure_ascii=False),
            )
            st.subheader("Avaliação")
            st.markdown(evaluation)
        except Exception as exc:
            st.error(f"Falha na simulação: {exc}")

with docs_tab:
    st.write(f"Chunks carregados: **{len(rag.chunks)}**")
    for chunk in rag.chunks:
        with st.expander(chunk.source):
            st.write(chunk.text)
