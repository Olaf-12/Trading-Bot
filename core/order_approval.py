"""Token de aprobación de un solo uso para comprar.

Los adaptadores (``ExecutionPort.buy``) exigen un ``OrderApproval`` válido. Solo
``ExecutionController`` lo emite, porque es el único módulo que importa la clave
privada ``_MINT_KEY``. Python no impide técnicamente saltarse la convención, pero:
- la GUI/estrategia no pueden comprar por accidente;
- ``tests/test_execution_demo.py`` verifica que ningún módulo fuera de ``core/``
  importe ``_MINT_KEY`` ni el módulo de ejecución.
"""

from __future__ import annotations

import threading
import time

_MINT_KEY = object()


class ApprovalError(PermissionError):
    pass


class OrderApproval:
    """Aprobación de compra de un solo uso, ligada a propuesta, precio y cuenta."""

    def __init__(self, *, key: object, client_order_id: str, proposal_id: str,
                 max_price: float, account_id: str, ttl_s: float = 10.0,
                 clock=time.time) -> None:
        if key is not _MINT_KEY:
            raise ApprovalError("Solo ExecutionController puede emitir aprobaciones")
        self.client_order_id = client_order_id
        self.proposal_id = proposal_id
        self.max_price = float(max_price)
        self.account_id = account_id
        self.ttl_s = ttl_s
        self._clock = clock
        self.issued_at = clock()
        self._used = False
        self._lock = threading.Lock()

    def consume(self, *, proposal_id: str, price: float, account_id: str) -> None:
        """Valida y marca como usada. Lanza ApprovalError si no corresponde."""
        with self._lock:
            if self._used:
                raise ApprovalError("Aprobación ya utilizada")
            if self._clock() - self.issued_at > self.ttl_s:
                raise ApprovalError("Aprobación caducada")
            if proposal_id != self.proposal_id:
                raise ApprovalError("La aprobación es para otra propuesta")
            if price > self.max_price + 1e-9:
                raise ApprovalError("Precio superior al aprobado")
            if account_id != self.account_id:
                raise ApprovalError("La aprobación es para otra cuenta")
            self._used = True

    def __repr__(self) -> str:
        return (f"OrderApproval(order={self.client_order_id}, proposal={self.proposal_id}, "
                f"max_price={self.max_price}, used={self._used})")
