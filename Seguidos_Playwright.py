import time
import random
import csv
import json
import os
from playwright.sync_api import sync_playwright
from dotenv import load_dotenv

# --- CONFIGURACIÓN ---
# achirov nayeli p2
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "chatpdf", ".env"))

def login_with_cookies(context, page):
    """Inicia sesión en Instagram usando las cookies desde .env."""
    raw_cookies = json.loads(os.getenv("INSTAGRAM_COOKIES"))

    cookies = []
    for c in raw_cookies:
        cookie = {
            'name': c['name'],
            'value': c['value'],
            'domain': c['domain'],
            'path': c['path'],
        }
        if c.get('expirationDate'):
            cookie['expires'] = c['expirationDate']
        if c.get('sameSite'):
            same_site_map = {'lax': 'Lax', 'no_restriction': 'None', 'strict': 'Strict'}
            cookie['sameSite'] = same_site_map.get(c['sameSite'], 'None')
        if c.get('secure'):
            cookie['secure'] = True
        if c.get('httpOnly'):
            cookie['httpOnly'] = True
        cookies.append(cookie)

    context.add_cookies(cookies)
    page.goto("https://www.instagram.com/")
    time.sleep(3)

    # Cerrar popup de notificaciones si aparece
    try:
        page.click('button:has-text("Ahora no")', timeout=5000)
    except:
        pass

    print(f"[OK] Sesión cargada con {len(cookies)} cookies")

def get_followers(page, target_username, limit=None):
    """Extrae la lista de seguidores de un usuario."""
    page.goto(f"https://www.instagram.com/{target_username}/")
    page.wait_for_selector('header', timeout=15000)
    time.sleep(2)

    # Click en "seguidores" / "followers"
    followers_link = page.locator(f'a[href="/{target_username}/followers/"]')
    if followers_link.count() == 0:
        followers_link = page.locator('a:has-text("seguidores"), a:has-text("followers")')

    followers_link.first.click()
    time.sleep(3)

    # Esperar a que cargue el modal/lista
    dialog = page.locator('div[role="dialog"]')
    dialog.wait_for(timeout=10000)
    time.sleep(2)

    # Buscar el contenedor scrolleable: el div dentro del dialog que tiene scroll
    scrollable = dialog.locator('div').filter(has=page.locator('div[role="progressbar"]')).first
    # Fallback: buscar cualquier div con overflow auto/scroll/hidden dentro del dialog
    if scrollable.count() == 0:
        scrollable = page.evaluate_handle('''() => {
            const dialog = document.querySelector('div[role="dialog"]');
            const divs = dialog.querySelectorAll('div');
            for (const div of divs) {
                if (div.scrollHeight > div.clientHeight && div.clientHeight > 100) {
                    return div;
                }
            }
            return null;
        }''')

    usernames = set()
    last_count = 0
    stale_rounds = 0

    print("Recopilando seguidores...")

    while True:
        # Extraer usernames: buscar los links dentro del dialog
        links = dialog.locator('a[role="link"]').all()
        for link in links:
            try:
                href = link.get_attribute('href', timeout=2000)
                if href and href.startswith('/') and href.count('/') == 2:
                    username = href.strip('/')
                    if username and ' ' not in username and len(username) > 1:
                        usernames.add(username)
            except:
                continue

        print(f" -> Recolectados: {len(usernames)}")

        if limit and len(usernames) >= limit:
            break

        # Verificar si dejó de cargar nuevos
        if len(usernames) == last_count:
            stale_rounds += 1
            if stale_rounds >= 20:
                print("No se encontraron más seguidores.")
                break
        else:
            stale_rounds = 0
            last_count = len(usernames)

        # Scroll incremental: avanza 600px en vez de saltar al fondo
        page.evaluate('''() => {
            const dialog = document.querySelector('div[role="dialog"]');
            const divs = dialog.querySelectorAll('div');
            for (const div of divs) {
                if (div.scrollHeight > div.clientHeight && div.clientHeight > 100) {
                    div.scrollTop += 600;
                    break;
                }
            }
        }''')

        time.sleep(2+random.uniform(2, 4))

    return list(usernames)

def get_user_details(page, username):
    """Obtiene detalles de un perfil visitando su página."""
    try:
        page.goto(f"https://www.instagram.com/{username}/", timeout=15000)
        page.wait_for_selector('header', timeout=10000)
        time.sleep(random.uniform(1, 2))

        # Extraer datos del header
        header = page.locator('header')

        # Nombre completo
        full_name = ""
        try:
            full_name = header.locator('span[class*="x1lliihq"]').first.inner_text(timeout=3000)
        except:
            pass

        # Bio
        biography = ""
        try:
            bio_el = page.locator('div[class*="x7a106z"] span').first
            biography = bio_el.inner_text(timeout=3000).replace('\n', ' ')
        except:
            pass

        # Métricas (seguidores, seguidos, publicaciones)
        metrics = page.locator('header li span[class*="x5n08af"], header li span span').all()
        followers = 0
        following = 0

        try:
            meta_texts = []
            for m in metrics:
                txt = m.get_attribute('title') or m.inner_text(timeout=2000)
                meta_texts.append(txt)

            if len(meta_texts) >= 3:
                followers = parse_number(meta_texts[1])
                following = parse_number(meta_texts[2])
        except:
            pass

        # Verificado
        is_verified = page.locator('svg[aria-label="Verified"], svg[aria-label="Verificado"]').count() > 0

        # Privado
        is_private = page.locator('h2:has-text("Esta cuenta es privada"), h2:has-text("This account is private")').count() > 0

        return {
            'username': username,
            'full_name': full_name,
            'biography': biography,
            'follower_count': followers,
            'following_count': following,
            'is_private': is_private,
            'is_verified': is_verified,
        }
    except Exception as e:
        print(f"   [x] Error en {username}: {e}")
        return None

def parse_number(text):
    """Convierte textos como '1.234' o '12,5 mil' a número."""
    text = text.strip().replace('.', '').replace(',', '.')
    text = text.lower()
    if 'mil' in text or 'k' in text:
        return int(float(text.replace('mil', '').replace('k', '').strip()) * 1000)
    elif 'mill' in text or 'm' in text:
        return int(float(text.replace('mill', '').replace('m', '').strip()) * 1000000)
    try:
        return int(text)
    except:
        return 0

def analyze_profile(user_data):
    """Clasifica el perfil según patrones."""
    tags = []
    bio = user_data['biography'].lower()
    followers = user_data['follower_count']
    following = user_data['following_count']

    keywords_business = ['ceo', 'founder', 'fundador', 'dueño', 'marketing', 'ventas', 'manager', 'gerente']
    if any(kw in bio for kw in keywords_business):
        tags.append("Lead Comercial")

    if followers > 100000:
        tags.append("Macro-Influencer")
    elif 5000 <= followers <= 100000:
        tags.append("Micro-Influencer")
    elif 1000 <= followers < 5000:
        tags.append("Nano-Influencer")

    if following > 1500 and followers < 100:
        tags.append("Posible Bot/Spam")

    if followers > 0 and following > 0 and (followers / following) > 2.0:
        tags.append("Alta Autoridad")

    if not tags:
        tags.append("Usuario Estándar")

    return ", ".join(tags)

def main():
    target_user = input("Usuario objetivo: ").strip()
    try:
        max_limit = int(input("Límite de seguidores a analizar (0 para todos): ") or 0)
    except:
        max_limit = 0
    if max_limit == 0:
        max_limit = None
    # "seguidores"
    filename = f"{target_user}_seguidores_playwright.csv"
    usernames_file = f"{target_user}_usernames_pendientes.json"
    keys = ['username', 'full_name', 'biography', 'analisis_patron', 'follower_count', 'following_count', 'is_private', 'is_verified']

    # Detectar perfiles ya procesados en ejecuciones anteriores
    already_done = set()
    if os.path.exists(filename):
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    already_done.add(row['username'])
            print(f"[REANUDAR] Se encontraron {len(already_done)} perfiles ya procesados en {filename}")
        except:
            pass

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(
            viewport={'width': 1280, 'height': 720},
            locale='es-ES'
        )
        page = context.new_page()

        # 1. Login con cookie
        login_with_cookies(context, page)

        # 2. Obtener lista de seguidores (o cargar de archivo previo)
        if os.path.exists(usernames_file) and already_done:
            with open(usernames_file, 'r', encoding='utf-8') as f:
                usernames = json.load(f)
            print(f"[REANUDAR] Lista de {len(usernames)} usernames cargada desde {usernames_file}")
        else:
            usernames = get_followers(page, target_user, limit=max_limit)

        if not usernames:
            print("No se encontraron seguidores.")
            browser.close()
            return

        # Guardar lista de usernames para poder reanudar si se cierra
        with open(usernames_file, 'w', encoding='utf-8') as f:
            json.dump(usernames, f)
        print(f"[OK] Lista de {len(usernames)} usernames guardada en {usernames_file}")

        # Filtrar los que ya fueron procesados
        pending = [u for u in usernames if u not in already_done]
        print(f"\nAnalizando {len(pending)} perfiles pendientes (de {len(usernames)} totales)...")

        # 3. Abrir CSV en modo append (o crear con headers si no existe)
        write_header = not os.path.exists(filename) or len(already_done) == 0
        csv_file = open(filename, 'a' if not write_header else 'w', newline='', encoding='utf-8')
        writer = csv.DictWriter(csv_file, fieldnames=keys)
        if write_header:
            writer.writeheader()
            csv_file.flush()

        count = len(already_done)
        total = len(usernames)

        try:
            for username in pending:
                count += 1
                print(f"   [{count}/{total}] {username} ...")

                details = get_user_details(page, username)
                if details:
                    details['analisis_patron'] = analyze_profile(details)
                    writer.writerow(details)
                    csv_file.flush()  # Forzar escritura a disco inmediatamente

                time.sleep(random.uniform(3, 6))
        except Exception as e:
            print(f"\n[!] Interrupcion: {e}")
            print(f"[OK] Progreso guardado en {filename} ({count} de {total})")
        finally:
            csv_file.close()

        # Limpiar archivo temporal si se completó todo
        if count >= total:
            if os.path.exists(usernames_file):
                os.remove(usernames_file)
            print(f"\n[ÉXITO] Completado! {count} perfiles guardados en: {filename}")
        else:
            print(f"\n[PARCIAL] {count}/{total} perfiles guardados en: {filename}")
            print(f"Ejecuta de nuevo el script con el mismo usuario para continuar.")

        browser.close()

if __name__ == "__main__":
    main()
