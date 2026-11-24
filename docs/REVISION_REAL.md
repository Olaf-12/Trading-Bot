# Fase 8 — Revisión previa al entorno Real

**Estado: Real BLOQUEADO.** `config.REAL_MODE_AVAILABLE = False` es una constante de
código. Ninguna variable de entorno, fichero, ajuste de SQLite ni control de la GUI
puede activarlo, y `tests/test_security.py` lo verifica en cada ejecución.

Esta fase **no se completa con código**. Según el documento de arquitectura (§8, §10),
Real solo se considera tras una revisión manual, legal y de seguridad. Este checklist
es la evidencia que debe existir **antes** de modificar esa constante.

## 1. Elegibilidad legal y regulatoria (responsable: titular de la cuenta)
- [ ] Confirmado por escrito con Deriv que la cuenta, la entidad contractual y el
      producto (opciones Rise/Fall de duración fija) están disponibles en el país de
      residencia real (Colombia, Venezuela u otro).
- [ ] Sin VPN ni datos de residencia falsos.
- [ ] Revisadas las obligaciones fiscales y cambiarias locales.
- [ ] Aceptado por escrito que un resultado histórico o en Demo no garantiza resultados
      futuros y que se puede perder todo el capital asignado.

## 2. Evidencia de Demo prolongado (Fase 7)
- [ ] `python main.py soak --symbol <SYM> --hours 168 --auto` (≥ 7 días) termina con
      `passed: true`.
- [ ] Cero reconciliaciones con discrepancias (`reconciliation_runs.ok = 1` en todas).
- [ ] Cero órdenes `unknown` sin resolver y cero contratos `unknown_remote`.
- [ ] Las paradas de emergencia y las pausas de riesgo se probaron a mano y funcionaron.
- [ ] Se probó a mano la reconexión (red cortada a mitad de un contrato abierto) y el
      contrato se liquidó o reconcilió correctamente.

## 3. Evidencia del modelo
- [ ] El modelo activo está `approved` en el registro (EV neto > 0 en walk-forward, mayoría
      de folds positivos, logloss mejor que el baseline).
- [ ] Backtest **fuera de muestra** (`python main.py backtest`) con EV neto > 0 y el payout
      real observado en Demo (no el 1.95 supuesto).
- [ ] El win rate en Demo prolongado supera el breakeven `1 / payout` con margen,
      sobre un número de trades estadísticamente significativo.
- [ ] Documentado el plan de reentrenamiento y de retirada del modelo si su rendimiento
      en vivo cae por debajo del breakeven.

## 4. Seguridad
- [ ] PAT de Real creado solo con el scope `trade` (sin `payment` ni `account_manage`).
- [ ] Tokens solo en `keyring` del SO; verificado que no aparecen en logs, SQLite ni Git.
- [ ] Revisión de código independiente de `core/execution_controller.py`,
      `broker/deriv_execution.py`, `broker/deriv_auth.py` y `risk/risk_manager.py`.
- [ ] Límites de riesgo de Real definidos por escrito: importe máximo, pérdida diaria y
      capital total asignado (dinero que se puede perder por completo).

## 5. Cambios técnicos necesarios para habilitar Real (NO hechos)
Hacerlos **solo** cuando las secciones 1–4 estén completas y firmadas:
1. Un `DerivRealAdapter` separado de `DerivDemoAdapter`, sin compartir configuración, que
   valide `account_type == "real"` y la ruta `/ws/real`.
2. Confirmación explícita **en cada sesión** (no persistida), con escritura del
   `account_id` y una frase de confirmación.
3. Límites de riesgo de Real independientes y más estrictos, que no se puedan relajar
   desde la GUI.
4. Cambiar `REAL_MODE_AVAILABLE` y actualizar `tests/test_security.py` en el mismo
   commit, con referencia a este checklist firmado.

Firma / fecha de la revisión: ______________________
