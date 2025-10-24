from typing import get_type_hints, Any, Dict, Type
from inspect import signature

class TypeHintCollector(type):
    """Metaclass that collects type hints from classes and their methods."""
    
    def __new__(mcs, name: str, bases: tuple, namespace: Dict[str, Any]) -> Type:
        # Create the class
        cls = super().__new__(mcs, name, bases, namespace)
        
        # Store type hints for the class itself
        cls.__type_hints__ = get_type_hints(cls)
        
        # Store type hints for all methods
        cls.__method_type_hints__ = {}
        for method_name, method in namespace.items():
            if callable(method):
                try:
                    # Get type hints for the method
                    method_hints = get_type_hints(method)
                    # Get parameter types from signature
                    sig = signature(method)
                    param_types = {
                        name: param.annotation 
                        for name, param in sig.parameters.items()
                    }
                    # Store both return type and parameter types
                    cls.__method_type_hints__[method_name] = {
                        'return': method_hints.get('return', Any),
                        'parameters': param_types
                    }
                except Exception as e:
                    print(f"Warning: Could not get type hints for method {method_name}: {e}")
        
        return cls

    @classmethod
    def get_class_type_hints(cls, target_class: Type) -> Dict[str, Any]:
        """Get all type hints for a class."""
        if not hasattr(target_class, '__type_hints__'):
            return {}
        return target_class.__type_hints__

    @classmethod
    def get_method_type_hints(cls, target_class: Type) -> Dict[str, Dict[str, Any]]:
        """Get all method type hints for a class."""
        if not hasattr(target_class, '__method_type_hints__'):
            return {}
        return target_class.__method_type_hints__ 