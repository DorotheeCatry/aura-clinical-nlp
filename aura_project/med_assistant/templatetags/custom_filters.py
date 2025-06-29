from django import template

register = template.Library()

@register.filter
def get_item(dictionary, key):
    """Permet d'accéder à un élément d'un dictionnaire avec une clé variable dans les templates"""
    return dictionary.get(key, [])

@register.filter
def mul(value, arg):
    """Multiplie deux valeurs"""
    try:
        return float(value) * float(arg)
    except (ValueError, TypeError):
        return 0

@register.filter
def div(value, arg):
    """Divise deux valeurs"""
    try:
        if float(arg) == 0:
            return 0
        return float(value) / float(arg)
    except (ValueError, TypeError):
        return 0

@register.filter
def subtract(value, arg):
    """Soustrait deux valeurs"""
    try:
        return float(value) - float(arg)
    except (ValueError, TypeError):
        return 0