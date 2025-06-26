from django import template

register = template.Library()

@register.filter
def get_item(dictionary, key):
    """Permet d'accéder à un élément d'un dictionnaire avec une clé variable dans les templates"""
    return dictionary.get(key, [])