from typing import Any, Callable, Dict, Optional, Union, List
import optax


def create_optimizer(
    optimizer_name: str = "adam",
    learning_rate: Union[float, Callable] = 1e-3,
    scheduler: Optional[str] = None,
    scheduler_params: Optional[Dict[str, Any]] = None,
    optimizer_params: Optional[Dict[str, Any]] = None,
    gradient_transforms: Optional[List[optax.GradientTransformation]] = None,
) -> optax.GradientTransformation:
    """
    Creates an optimizer with optional learning rate schedule and gradient transformations.

    Args:
        optimizer_name: The name of the optimizer to use (e.g., "adam", "adamw", "sgd", "lion").
        learning_rate: The learning rate or a learning rate schedule function.
        scheduler: The name of the learning rate scheduler to use (e.g., "warmup_cosine_decay", "exponential_decay").
        scheduler_params: Parameters for the learning rate scheduler.
        optimizer_params: Additional parameters for the optimizer.
        gradient_transforms: List of additional gradient transformations to apply.

    Returns:
        An optax.GradientTransformation object representing the optimizer.
    """
    if scheduler:
        learning_rate = _create_scheduler(learning_rate, scheduler, scheduler_params or {})

    optimizer_params = optimizer_params or {}
    optimizer = _create_base_optimizer(optimizer_name, learning_rate, optimizer_params)

    if gradient_transforms:
        optimizer = optax.chain(*gradient_transforms, optimizer)

    return optimizer

def _create_scheduler(
    learning_rate: Union[float, Callable],
    scheduler: str,
    params: Dict[str, Any]
) -> Callable:
    if isinstance(learning_rate, float):
        if scheduler == "warmup_cosine_decay":
            return optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=learning_rate,
                **params
            )
        elif scheduler == "exponential_decay":
            return optax.exponential_decay(
                init_value=learning_rate,
                **params
            )
        elif scheduler == "cosine_decay":
            return optax.cosine_decay_schedule(
                init_value=learning_rate,
                **params
            )
        elif scheduler == "polynomial":
            return optax.polynomial_schedule(
                init_value=learning_rate,
                **params
            )
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler}")
    return learning_rate

def _create_base_optimizer(
    optimizer_name: str,
    learning_rate: Union[float, Callable],
    params: Dict[str, Any]
) -> optax.GradientTransformation:
    if optimizer_name == "adam":
        return optax.adam(learning_rate, **params)
    elif optimizer_name == "adamw":
        return optax.adamw(learning_rate, **params)
    elif optimizer_name == "sgd":
        return optax.sgd(learning_rate, **params)
    elif optimizer_name == "rmsprop":
        return optax.rmsprop(learning_rate, **params)
    elif optimizer_name == "adagrad":
        return optax.adagrad(learning_rate, **params)
    elif optimizer_name == "lion":
        return optax.lion(learning_rate, **params)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
