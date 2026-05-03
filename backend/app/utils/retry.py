"""
Mécanisme de nouvelle tentative pour les appels API
Gère la logique de nouvelle tentative pour les appels API externes comme le LLM
"""

import time
import random
import functools
from typing import Callable, Any, Optional, Type, Tuple
from ..utils.logger import get_logger

logger = get_logger('mirofish.retry')


def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 30.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None
):
    """
    Décorateur de nouvelle tentative avec intervalle exponentiel

    Args:
        max_retries: Nombre maximal de tentatives
        initial_delay: Délai initial (secondes)
        max_delay: Délai maximal (secondes)
        backoff_factor: Facteur d'intervalle
        jitter: Ajouter ou non une gigue aléatoire
        exceptions: Types d'exceptions à retenter
        on_retry: Rappel lors de la nouvelle tentative (exception, nombre de tentatives)

    Utilisation :
        @retry_with_backoff(max_retries=3)
        def call_llm_api():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            delay = initial_delay

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)

                except exceptions as e:
                    last_exception = e

                    if attempt == max_retries:
                        logger.error(f"La fonction {func.__name__} a encore échoué après {max_retries} tentatives : {str(e)}")
                        raise

                    # Calculer le délai
                    current_delay = min(delay, max_delay)
                    if jitter:
                        current_delay = current_delay * (0.5 + random.random())

                    logger.warning(
                        f"La fonction {func.__name__} a échoué à la tentative {attempt + 1} : {str(e)}, "
                        f"nouvelle tentative dans {current_delay:.1f} secondes..."
                    )

                    if on_retry:
                        on_retry(e, attempt + 1)

                    time.sleep(current_delay)
                    delay *= backoff_factor

            raise last_exception

        return wrapper
    return decorator


def retry_with_backoff_async(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 30.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None
):
    """
    Version asynchrone du décorateur de nouvelle tentative
    """
    import asyncio
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            delay = initial_delay

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)

                except exceptions as e:
                    last_exception = e

                    if attempt == max_retries:
                        logger.error(f"La fonction asynchrone {func.__name__} a encore échoué après {max_retries} tentatives : {str(e)}")
                        raise

                    current_delay = min(delay, max_delay)
                    if jitter:
                        current_delay = current_delay * (0.5 + random.random())

                    logger.warning(
                        f"La fonction asynchrone {func.__name__} a échoué à la tentative {attempt + 1} : {str(e)}, "
                        f"nouvelle tentative dans {current_delay:.1f} secondes..."
                    )

                    if on_retry:
                        on_retry(e, attempt + 1)

                    await asyncio.sleep(current_delay)
                    delay *= backoff_factor

            raise last_exception

        return wrapper
    return decorator


class RetryableAPIClient:
    """
    Client API avec mécanisme de nouvelle tentative
    """

    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 30.0,
        backoff_factor: float = 2.0
    ):
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor

    def call_with_retry(
        self,
        func: Callable,
        *args,
        exceptions: Tuple[Type[Exception], ...] = (Exception,),
        **kwargs
    ) -> Any:
        """
        Exécuter un appel de fonction avec nouvelle tentative en cas d'échec

        Args:
            func: Fonction à appeler
            *args: Arguments de la fonction
            exceptions: Types d'exceptions à retenter
            **kwargs: Arguments nommés de la fonction

        Returns:
            Valeur de retour de la fonction
        """
        last_exception = None
        delay = self.initial_delay

        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)

            except exceptions as e:
                last_exception = e

                if attempt == self.max_retries:
                    logger.error(f"L'appel API a encore échoué après {self.max_retries} tentatives : {str(e)}")
                    raise

                current_delay = min(delay, self.max_delay)
                current_delay = current_delay * (0.5 + random.random())

                logger.warning(
                    f"L'appel API a échoué à la tentative {attempt + 1} : {str(e)}, "
                    f"nouvelle tentative dans {current_delay:.1f} secondes..."
                )

                time.sleep(current_delay)
                delay *= self.backoff_factor

        raise last_exception

    def call_batch_with_retry(
        self,
        items: list,
        process_func: Callable,
        exceptions: Tuple[Type[Exception], ...] = (Exception,),
        continue_on_failure: bool = True
    ) -> Tuple[list, list]:
        """
        Appel par lot avec nouvelle tentative individuelle pour chaque élément échoué

        Args:
            items: Liste des éléments à traiter
            process_func: Fonction de traitement, accepte un seul élément en paramètre
            exceptions: Types d'exceptions à retenter
            continue_on_failure: Continuer ou non le traitement des autres éléments après un échec

        Returns:
            (liste des résultats réussis, liste des éléments échoués)
        """
        results = []
        failures = []

        for idx, item in enumerate(items):
            try:
                result = self.call_with_retry(
                    process_func,
                    item,
                    exceptions=exceptions
                )
                results.append(result)

            except Exception as e:
                logger.error(f"Échec du traitement de l'élément {idx + 1} : {str(e)}")
                failures.append({
                    "index": idx,
                    "item": item,
                    "error": str(e)
                })

                if not continue_on_failure:
                    raise

        return results, failures
