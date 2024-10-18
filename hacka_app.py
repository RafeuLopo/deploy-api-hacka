from flask import Blueprint, request, jsonify, Flask, session
from flask_cors import CORS
import pandas as pd
import requests
import os

API_KEY = os.getenv('API_KEY')
BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-3.5-turbo-0125"

api = Blueprint('api', __name__)

def load_faq(file_path):
    try:
        faq_df = pd.read_csv(file_path, encoding='utf-8')

        if 'pergunta' not in faq_df.columns or 'resposta' not in faq_df.columns:
            raise ValueError("O arquivo deve conter as colunas 'pergunta' e 'resposta'.")

        faq_dict = dict(zip(faq_df['pergunta'].str.lower(), faq_df['resposta']))
        return faq_dict

    except FileNotFoundError:
        print(f"Arquivo {file_path} não encontrado.")
    except pd.errors.EmptyDataError:
        print("O arquivo CSV está vazio.")
    except Exception as e:
        print(f"Ocorreu um erro ao ler o arquivo FAQ: {e}")
    return {}


def load_customer_data(file_path):
    try:
        customer_df = pd.read_csv(file_path, encoding='utf-8')
        return customer_df
    except FileNotFoundError:
        print(f"Arquivo {file_path} não encontrado.")
    except pd.errors.EmptyDataError:
        print("O arquivo CSV está vazio.")
    except Exception as e:
        print(f"Ocorreu um erro ao ler o arquivo de clientes: {e}")
    return pd.DataFrame()

def identify_persona(customer):
    age = customer['Idade']
    senior_citizen = customer['SeniorCitizen']
    name = customer['Nome']
    gender = customer['gender']
    
    prompt = f"""
    Com base nas seguintes informações do cliente, identifique a persona:

    Nome: {name}
    Idade: {age}
    SeniorCitizen: {senior_citizen}
    Gênero: {gender}
    
    As personas são:
    - Persona Dona Maria: Uma mulher de 70 anos com dificuldade para enxergar o aplicativo, precisa de uma comunicação clara e amigável.
    - Persona Nicolas: Um jovem de 20 anos que gosta de explorar todas as funções do aplicativo, prefere uma comunicação casual e dinâmica.
    
    Responda com uma descrição da persona (Dona Maria ou Nicolas).
    """
    
    return call_openai_api(prompt)


def greetings_persona(customer):
    persona = identify_persona(customer)

    prompt = f"""
    Baseado nas informações abaixo, gere apenas uma saudação adequada e amigável para o cliente. Não mencione a persona explicitamente.

    Nome: {customer['Nome']}
    Persona: {persona}

    Instruções:
    - Para um cliente mais velho (como Dona Maria), use uma comunicação clara, acolhedora e formal.
    - Para um cliente jovem (como Nicolas), use uma comunicação casual, amigável e dinâmica.

    Não inclua informações da persona na saudação, apenas o tom correto e a saudação amigável.

    Utilize sempre o nome do Cliente, não o da Persona.

    """
    return call_openai_api(prompt)


def analyze_spending_profile(customer):
    prompt = f"""
    Analisando as seguintes informações do cliente, identifique seu perfil de compra/gasto:

    ID: {customer['customerID']}
    Nome: {customer['Nome']}
    Idade: {customer['Idade']}
    Renda: {customer['RendaReais']}
    Partner: {customer['Partner']}
    Dependents: {customer['Dependents']}
    PhoneService: {customer['PhoneService']}
    InternetService: {customer['InternetService']}
    MonthlyCharges: {customer['MonthlyCharges']}
    DeviceProtection: {customer['DeviceProtection']}
    StreamingTV: {customer['StreamingTV']}
    StreamingMovies: {customer['StreamingMovies']}
    Contract: {customer['Contract']}
    PaymentMethod: {customer['PaymentMethod']}
    TotalCharges: {customer['TotalCharges']}
    TVUsageHours: {customer['TVUsageHours']}
    InternetUsageGB: {customer['InternetUsageGB']}
    PhoneUsageHours: {customer['PhoneUsageHours']}
    PreviousPurchases: {customer['PreviousPurchases']}
    RendaReais: {customer['RendaReais']}

    Responda com uma descrição do perfil de compra/gasto do cliente.
    """

    return call_openai_api(prompt)


def suggest_offer(customer):
    spending_profile_analysis = analyze_spending_profile(customer)
    
    prompt = f"""
    Baseado nas seguintes informações, sugira a melhor oferta para o cliente:
    
    Análise do perfil de gastos: {spending_profile_analysis}

    Responda com a melhor oferta para esse cliente. Nota-se que o valor total de TotalCharges
    não deve ser alterado de maneira significante. Além disso, o valor deve ser compatível com a RendaReais do cliente.
    """

    return call_openai_api(prompt)

def notification_offer(customer):
    spending_profile_analysis = analyze_spending_profile(customer)
    
    prompt = f"""
    Baseado nas seguintes informações, gere uma notificação de oferta personalizada para o cliente:
    
    Análise do perfil de gastos: {spending_profile_analysis}
    
    Estruture a notificação da seguinte forma:
    - Título chamativo e breve que destaque a oferta. Faça uma pergunta envolvente no título.
    - Uma frase introdutória e chamativa que faça o cliente se interessar pela oferta. Evite parecer um scam.
    - Descrição clara e detalhada da oferta, explicando como ela atende ao perfil do cliente.
    - Um fechamento convidando o cliente a aproveitar a oferta.
    - Busque textos mais curtos e evite a palavra "conservador".

    Certifique-se de que o valor total de "TotalCharges" não seja alterado de maneira significante e que o valor seja compatível com a renda mensal do cliente ({customer['RendaReais']}).
    
    Responda apenas com a notificação formatada.
    """

    return call_openai_api(prompt)


def generate_payment_prompt(customer):
    """
    Gera o prompt com base nos dados do cliente para análise de status de pagamento.
    """
    prompt = f"""
    Você é um assistente virtual que ajuda a analisar a situação de pagamento dos clientes e envia lembretes proativos para evitar interrupções no serviço.

    Abaixo estão as informações do cliente:

    - Nome: {customer['Nome']}
    - Método de Pagamento: {customer['PaymentMethod']}
    - Data de Expiração do Cartão: {customer['CardExpiryDate']}
    - Última Data de Pagamento: {customer['LastPaymentDate']}
    - Status da Assinatura: {customer['SubscriptionStatus']}
    - Suspeita de Fraude: {customer['FraudSuspected']}

    A tarefa é identificar possíveis problemas com o pagamento do cliente, como:
    - Cartão expirado
    - Assinatura não renovada ou suspensa
    - Falta de pagamento recente
    - Suspeita de fraude

    Se houver algum problema, envie uma notificação clara e amigável ao cliente para corrigir a situação. Se tudo estiver em ordem, informe ao cliente que sua conta está em dia.

    Responda com uma mensagem clara e proativa.
    """

    return prompt


def analyze_payment_with_gpt(customer):
    """
    Usa a IA Generativa para analisar o status de pagamento e gerar uma resposta.
    """
    prompt = generate_payment_prompt(customer)

    response = call_openai_api(prompt)

    return response


def notify_customer_payment_status_with_gpt(customer):
    """
    Usa a IA para gerar a notificação do status de pagamento e enviá-la ao cliente.
    """
    payment_notification = analyze_payment_with_gpt(customer)

    return payment_notification


def call_openai_api(prompt):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": DEFAULT_MODEL,
        "messages": [{"role": "user", "content": prompt}]
    }

    try:
        response = requests.post(f"{BASE_URL}/chat/completions", headers=headers, json=data)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content'].strip()
    except requests.RequestException as e:
        print(f"Erro ao acessar a API: {e}")
        return "Desculpe, não consegui gerar uma resposta no momento."
    

@api.route('/persona', methods=['POST'])
def persona():
    customer_data = request.json
    try:
        persona = identify_persona(customer_data)
        return jsonify({"persona": persona}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api.route('/greeting', methods=['POST'])
def greeting():
    customer_data = request.json
    try:
        greeting_msg = greetings_persona(customer_data)
        return jsonify({"greeting": greeting_msg}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api.route('/offer', methods=['POST'])
def offer():
    customer_data = request.json
    try:
        offer_msg = notification_offer(customer_data)
        return jsonify({"offer": offer_msg}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api.route('/payment-status', methods=['POST'])
def payment_status():
    customer_data = request.json
    try:
        payment_msg = notify_customer_payment_status_with_gpt(customer_data)
        return jsonify({"payment_notification": payment_msg}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@api.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get("user_input")
    customer_data = data.get("customer_data")

    if not user_input or not customer_data:
        return jsonify({"error": "user_input e customer_data são necessários."}), 400

    if 'chat_history' not in session:
        session['chat_history'] = []

    try:
        if len(session['chat_history']) == 0:
            greeting = greetings_persona(customer_data)
            session['chat_history'].append({"role": "assistant", "content": greeting})

        session['chat_history'].append({"role": "user", "content": user_input})

        assistant_response = call_openai_api_with_history(session['chat_history'])

        session['chat_history'].append({"role": "assistant", "content": assistant_response})

        return jsonify({"response": assistant_response}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def call_openai_api_with_history(messages):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": DEFAULT_MODEL,
        "messages": messages
    }

    try:
        response = requests.post(f"{BASE_URL}/chat/completions", headers=headers, json=data)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content'].strip()
    except requests.RequestException as e:
        print(f"Erro ao acessar a API: {e}")
        return "Desculpe, não consegui gerar uma resposta no momento."


@api.route('/')
def home():
    print('Hello, World')


def create_app():
    app = Flask(__name__)
    
    CORS(app)
    app.secret_key = API_KEY
    app.config['PERMANENT_SESSION_LIFETIME'] = SESSION_LIFETIME
    app.register_blueprint(api, url_prefix="/api")

    return app

app = create_app()
