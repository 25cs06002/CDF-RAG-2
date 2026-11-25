-- Ensure prose-heavy manuscript tables fit the available line width.
-- Equal widths are safe for long prose columns, with shorter leading labels.
function Table(t)
  local n = #t.colspecs
  if n > 0 then
    for i = 1, n do
      local w = 1 / n
      if n >= 4 then
        w = (i == 1) and 0.16 or 0.84 / (n - 1)
      end
      t.colspecs[i] = {t.colspecs[i][1], w}
    end
  end
  return t
end
