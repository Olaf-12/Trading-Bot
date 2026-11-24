# TradingBotML

Bot de trading con ML para **opciones de duración fija (Rise/Fall)** sobre la API
oficial de Deriv. Python 3.12 + PySide6. **Solo Demo**: el entorno Real está bloqueado
por código (ver `docs/REVISION_REAL.md`).

> Guía de arquitectura y prototipado, no asesoramiento financiero. Un resultado
> histórico o en Demo no garantiza resultados futuros.

## Instalación
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Credenciales (solo para Demo): crea un PAT con el scope **`trade`** y guárdalo en el
almacén seguro del sistema (nunca en código, `.env` versionado ni SQLite):
```bash
python -c "import keyring; keyring.set_password('TradingBotML', 'DERIV_TOKEN', input('PAT: '))"
python -c "import keyring; keyring.set_password('TradingBotML', 'DERIV_APP_ID', input('App ID: '))"
```

## Flujo de trabajo recomendado
| Paso | Comando | Fase |
|---|---|---|
| Ver símbolos disponibles | `python main.py symbols` | 1 |
| Ver ticks en vivo | `python main.py ticks --symbol R_100 --seconds 60` | 1 |
| Comprobar cuenta y saldo Demo | `python main.py balance` | 2 |
| Descargar histórico | `python main.py download --symbol R_100 --count 5000 --pages 20` | 5 |
| Entrenar (walk-forward) | `python main.py train --symbol R_100 --horizon 5 --activate` | 5 |
| Backtest fuera de muestra | `python main.py backtest --symbol R_100` | 4 |
| GUI con cuenta Demo | `python main.py gui --demo` | 6 |
| Demo prolongado headless | `python main.py soak --symbol R_100 --hours 168 --auto` | 7 |

`--activate` solo activa el modelo si queda **aprobado**: EV neto de payout > 0 en
walk-forward, mayoría de folds positivos, logloss mejor que el baseline y suficientes
trades. Si no, se registra pero no se puede usar para operar.

## Arquitectura
```
interface/  GUI PySide6 -> solo comandos al TradingEngine (nunca al broker)
core/       trading_engine (hilos: trabajo + scheduler), execution_controller (ÚNICO
            punto que aprueba órdenes), order_approval (token de un solo uso),
            reconciliation, backtester, soak, bootstrap, event_manager
broker/     deriv_ws_client (bucle receptor central, enrutado por req_id/subscription.id/
            msg_type, reconexión), deriv_auth (PAT -> OTP), deriv_adapter, deriv_market,
            deriv_execution, simulated_broker
market/     validación de ticks, velas, indicadores causales
machine_learning/  features (ventana finita), dataset, train (walk-forward purgado),
            evaluate (neto de payout), predict (carga con verificación SHA-256)
strategy/   señales, estrategia de duración fija (umbral > breakeven)
risk/       límites, pausas, exposición, parada de emergencia, sizing (sin martingala)
database/   SQLite con migraciones: ticks, órdenes, trades, predicciones, modelos, auditoría
```

## Garantías de seguridad (verificadas por `tests/`)
- Demo por defecto; Real imposible de seleccionar (constante de código, no ajuste).
- `account_id` y tipo de cuenta validados por REST y por `balance.loginid` al conectar,
  tras cada reconexión y antes de cada compra; cada contrato comprueba `account_id`.
- OTP nuevo en cada conexión; la URL OTP y el token nunca se registran ni persisten.
- `buy` exige un `OrderApproval` del ExecutionController; nunca se reintenta: un timeout
  deja la orden `unknown` y la reconciliación la resuelve contra `portfolio`/`profit_table`.
- Contratos ajenos al bot en la cuenta → parada de emergencia.

## Pruebas
```bash
QT_QPA_PLATFORM=offscreen python -m pytest -q
```
Las pruebas no usan red ni credenciales: Deriv se simula en memoria.

## Qué NO está verificado contra el servidor real
El desarrollo se hizo sin acceso de red a Deriv. Los nombres de campos se contrastaron
con `deriv-com/deriv-api-schemas`, pero falta comprobar en vivo: que el endpoint nuevo
use esos mensajes, el host del OTP, que `balance.loginid` coincida con `account_id`, el
payout real por símbolo y duración, y los límites de peticiones.
