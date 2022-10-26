# Astea

Code traversal engine {% .font18 .margint12 %}

An advanced helper module for parsing and traversing any arbitrary python codebase, think of this like a programatic lookup that you perform in a full fledged IDE.

{{ tea.find("Astea")[0].docstring() }}

## class `Astea` {% .margint8 %}
{{ doc_str_to_mdx(tea.find('Astea.__init__')[0].docstring()) }}

{!% for x in tea.find('Astea')[0].find(types = IndexTypes.FUNCTION) %!}
{{ doc_str_to_mdx(x.docstring(), x.name) }}
{!% endfor %!}

