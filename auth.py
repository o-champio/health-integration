import json
import os
from requests_oauthlib import OAuth2Session
import config # Importa o seu ficheiro config.py

os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

def save_token(token):
    """Guarda o token num ficheiro JSON."""
    with open(config.TOKEN_FILE, 'w') as f:
        json.dump(token, f)
    print("Token salvo/atualizado localmente.")

def get_oura_session():
    """Configura a sessao OAuth2."""
    token = None
    if os.path.exists(config.TOKEN_FILE):
        with open(config.TOKEN_FILE, 'r') as f:
            token = json.load(f)

    # Usa as credenciais que vieram do config.py (OS)
    extra = {'client_id': config.OURA_CLIENT_ID, 'client_secret': config.OURA_CLIENT_SECRET}
    
    # O parametro scope deve ser passado aqui
    return OAuth2Session(
        config.OURA_CLIENT_ID,
        scope=config.SCOPES,
        token=token,
        auto_refresh_kwargs=extra,
        auto_refresh_url=config.TOKEN_URL,
        token_updater=save_token,
        redirect_uri=config.OURA_REDIRECT_URL
    )

def perform_auth_handshake(oura):
    """Gera a URL, pega a resposta do utilizador e troca pelo token."""
    # O parametro scope foi removido desta chamada
    authorization_url, state = oura.authorization_url(config.AUTH_URL)
    print(f"\n1. Aceda para autorizar:\n{authorization_url}")
    
    redirect_response = input("\n2. Cole a URL completa do redirecionamento aqui: ")
    
    token = oura.fetch_token(
        config.TOKEN_URL,
        authorization_response=redirect_response,
        client_secret=config.OURA_CLIENT_SECRET
    )
    save_token(token)
    return token

def main():
    # Validacao rapida pra ver se o OS puxou as variaveis direito
    if not config.OURA_CLIENT_ID or not config.OURA_CLIENT_SECRET:
        print("Erro: OURA_CLIENT_ID ou OURA_CLIENT_SECRET nao encontrados nas variaveis de ambiente.")
        return

    oura = get_oura_session()

    if not oura.token:
        print("Nenhum token encontrado. Iniciando autenticacao...")
        perform_auth_handshake(oura)

    print("\nTestando conexao com a API...")
    url_pessoal = f"{config.BASE_URL}personal_info"
    
    response = oura.get(url_pessoal)

    if response.status_code == 200:
        data = response.json()
        print("Sucesso!")
        print(f"Utilizador ID: {data.get('id')}")
    else:
        print(f"Erro {response.status_code}: {response.text}")

if __name__ == "__main__":
    main()