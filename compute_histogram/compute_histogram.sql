with s as (select sum(count) as total from histo),
     t as (select value, count, floor( ((sum(count) OVER (ORDER BY value))/s.total) * 10) as pct from histo, s)
select pct * 10 as percentile, min(value), max(value), sum(count) as count from t group by pct order by min(value)