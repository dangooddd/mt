ENTRY
  { note }
  { }
  { }

FUNCTION {begin.bib}
{
  "\begin{thebibliography}{99}" write$ newline$
}

FUNCTION {end.bib}
{
  "\end{thebibliography}" write$ newline$
}

FUNCTION {output.bibitem}
{
  newline$
  "\bibitem{" write$
  cite$ write$
  "}" write$ newline$
  note write$ newline$
}

FUNCTION {misc} { output.bibitem }
FUNCTION {default.type} { output.bibitem }

READ
EXECUTE {begin.bib}
ITERATE {call.type$}
EXECUTE {end.bib}
