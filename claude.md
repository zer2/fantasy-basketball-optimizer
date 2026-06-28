- Do not add "safe" defaults to functions or operations when there is no valid reason for a correct value to be provided. It is better to fail noisily than to contradict the logical flow quietly 
- Function names should be descriptive verbs, both in python and javascript 
- For this codebase, we are trying to build a JS version of a codebase that already works using streamlit. For the most part, we do not want to contradict how the streamlit code works in the backend (though I might make some changes)
- Function headers should always have linebreaks between parameter inputs unless they are extremely simple. Commas should always be at the beginning of each parameter line, not the end 
- Wherever possible (e.g. not the stat styler functions) CSS should be derived from styles.css
- Parameter should be framed in the positive. E.g. 'usemargin' instead of 'nomargin'
- Variable names should be descriptive. Except in one-liners like lamdba functions, let's never use any abbreviations 
- No circular dependenies or clumsy fixes for circular dependencies, like loading functions as parameters from other functions 

Do not push to github unless I explicitly tell you to