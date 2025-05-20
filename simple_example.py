import PyPDF2
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv
import os

# Carregar variáveis do arquivo .env
load_dotenv()

# Configuração do tema do Streamlit
st.set_page_config(
    page_title="Chatbot de Pré-Atendimento Dermatológico",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Estilização CSS personalizada
st.markdown(
    """
    <style>
    /* Estilo geral */
    .main {
        background-color: #f0f4f8;
        padding: 20px;
        border-radius: 10px;
    }
    /* Título */
    h1 {
        color: #2c3e50;
        font-family: 'Arial', sans-serif;
        text-align: center;
        font-size: 2.5em;
    }
    /* Botão */
    .stButton>button {
        background-color: #3498db;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 1.1em;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #2980b9;
    }
    /* Campo de texto */
    .stTextInput>div>input {
        border: 2px solid #3498db;
        border-radius: 8px;
        padding: 10px;
        font-size: 1em;
    }
    /* Mensagens de sucesso e erro */
    .stSuccess {
        background-color: #d4edda;
        color: #155724;
        border-radius: 8px;
        padding: 10px;
    }
    .stWarning {
        background-color: #fff3cd;
        color: #856404;
        border-radius: 8px;
        padding: 10px;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# Função para extrair texto de um PDF (com cache)
@st.cache_data
def extract_text_from_pdf(pdf_path):
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
            return text
    except Exception as e:
        st.error(f"Erro ao ler o PDF: {e}")
        return ""


# Função para inicializar o modelo (com cache)
@st.cache_resource
def initialize_model():
    llm = ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="gpt-3.5-turbo",
    )
    prompt_template = PromptTemplate(
        input_variables=["patient_data", "diagnosis_text"],
        template="""
        Com base nas informações fornecidas pelo paciente e no texto de referência sobre diagnósticos dermatológicos, sugira possíveis diagnósticos.

        **Informações do Paciente**:
        {patient_data}

        **Texto de Referência**:
        {diagnosis_text}

        Forneça uma lista de possíveis diagnósticos com uma breve explicação para cada um, em formato de relatório médico claro e organizado.
        """,
    )
    return prompt_template | llm


# Caminho para o arquivo PDF
pdf_path = "/Users/lucaspinheiro/Documents/VSCodeFiles/Codigos/Langchain_Tests/diagnosticos_dermatologia.pdf"

# Carregar dados de diagnóstico
diagnosis_text = extract_text_from_pdf(pdf_path)
if not diagnosis_text:
    st.warning(
        "Nenhum texto foi extraído do PDF. Verifique o arquivo e tente novamente."
    )

# Inicializar o modelo
chain = initialize_model()


# Função para analisar respostas
def analyze_responses(patient_data, diagnosis_text):
    try:
        response = chain.invoke(
            {"patient_data": patient_data, "diagnosis_text": diagnosis_text[:5000]}
        )
        return response.content.split("\n")
    except Exception as e:
        return [f"Erro ao analisar os dados: {e}"]


# Protocolo de perguntas
questions = [
    {"key": "name", "text": "Qual é o seu nome?"},
    {
        "key": "symptoms",
        "text": "Descreva seus sintomas principais (ex.: vermelhidão, coceira, descamação):",
    },
    {
        "key": "location",
        "text": "Onde os sintomas estão localizados (ex.: braços, rosto, pernas)?",
    },
    {
        "key": "duration",
        "text": "Há quanto tempo você está com esses sintomas (ex.: dias, semanas, meses)?",
    },
    {
        "key": "intensity",
        "text": "Qual é a intensidade dos sintomas (leve, moderada ou grave)?",
    },
    {
        "key": "triggers",
        "text": "Algum fator desencadeia ou agrava os sintomas (ex.: exposição ao sol, produtos químicos)?",
    },
    {
        "key": "history",
        "text": "Você tem algum histórico médico relevante (ex.: alergias, psoríase, eczema)?",
    },
    {
        "key": "medications",
        "text": "Está usando algum medicamento ou tratamento para os sintomas (ex.: cremes, antibióticos)?",
    },
]

# Inicializar estado da sessão
if "step" not in st.session_state:
    st.session_state.step = 0
if "responses" not in st.session_state:
    st.session_state.responses = {}
if "diagnoses" not in st.session_state:
    st.session_state.diagnoses = None
if "patient_data" not in st.session_state:
    st.session_state.patient_data = None

# Interface do Streamlit
st.title("🩺 Chatbot de Pré-Atendimento Dermatológico")

# Introdução
st.markdown(
    """
    Bem-vindo ao **Chatbot de Pré-Atendimento Dermatológico**! Responda às perguntas abaixo para que possamos coletar informações sobre seus sintomas. Nosso sistema seguirá um protocolo fixo e sugerirá possíveis diagnósticos com base em um guia dermatológico.
""",
    unsafe_allow_html=True,
)

# Divisor
st.markdown("---")

# Formulário para perguntas sequenciais
if st.session_state.step < len(questions):
    question = questions[st.session_state.step]
    with st.form(key=f"protocol_form_{st.session_state.step}"):
        # Usar chave única para o input, combinando key e step
        response = st.text_input(
            question["text"],
            placeholder="Digite sua resposta aqui...",
            key=f"{question['key']}_{st.session_state.step}",
        )
        submit_button = st.form_submit_button("➡️ Próxima Pergunta")

        if submit_button and response:
            st.session_state.responses[question["key"]] = response
            st.session_state.step += 1
        elif submit_button and not response:
            st.warning("Por favor, responda a pergunta antes de continuar.")
else:
    with st.form(key="final_form"):
        submit_button = st.form_submit_button("🔍 Finalizar e Analisar")
        if submit_button and st.session_state.responses:
            with st.spinner("Analisando seus dados..."):
                patient_data = "\n".join(
                    [
                        f"{key.capitalize()}: {value}"
                        for key, value in st.session_state.responses.items()
                    ]
                )
                st.session_state.patient_data = patient_data
                st.session_state.diagnoses = analyze_responses(
                    patient_data, diagnosis_text
                )
                st.session_state.step = 0
                st.session_state.responses = {}
        elif submit_button and not st.session_state.responses:
            st.warning("Nenhuma resposta fornecida. Por favor, reinicie o protocolo.")

# Exibir relatório fora do formulário
if st.session_state.diagnoses:
    st.success("Análise concluída!")
    st.subheader("Relatório de Pré-Atendimento")
    st.markdown("### Informações do Paciente")
    st.markdown(st.session_state.patient_data)
    st.markdown("### Possíveis Diagnósticos")
    for diagnosis in st.session_state.diagnoses:
        if diagnosis.strip():
            st.markdown(f"- {diagnosis}")
    # Botão para baixar relatório
    report = (
        f"Relatório de Pré-Atendimento Dermatológico\n\nInformações do Paciente:\n{st.session_state.patient_data}\n\nPossíveis Diagnósticos:\n"
        + "\n".join(st.session_state.diagnoses)
    )
    st.download_button(
        label="📄 Baixar Relatório",
        data=report,
        file_name="relatorio_dermatologia.txt",
        mime="text/plain",
    )

# Sidebar com informações adicionais
with st.sidebar:
    st.header("ℹ️ Sobre o Chatbot")
    st.markdown("""
        Este chatbot segue um protocolo fixo de pré-atendimento dermatológico, coletando informações estruturadas para sugerir possíveis diagnósticos com base em um documento de referência.

        **Nota**: Este é um sistema de apoio e não substitui a consulta com um dermatologista profissional.
    """)
    st.markdown("---")
    st.subheader("Dicas para melhores resultados")
    st.markdown("""
        - Responda às perguntas com o maior nível de detalhe possível.
        - Certifique-se de que o PDF de referência contém informações claras e legíveis.
        - Consulte um médico para um diagnóstico definitivo.
    """)
