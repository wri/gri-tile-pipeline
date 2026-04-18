-- TTC coverage stats per project for terrafund-landscapes
-- ttc column is MAP(year -> percent_cover)
-- "correct year" = map contains key YEAR(plantstart) - 1
--
-- Usage: duckdb < scripts/ttc_coverage_stats.sql
-- Or:   duckdb < scripts/ttc_coverage_stats.sql > ttc_coverage_terrafund.csv
COPY (
    SELECT
        short_name,
        COUNT(*) AS total_polys,
        COUNT(*) FILTER (WHERE ttc IS NOT NULL AND cardinality(ttc) > 0)
            AS with_ttc,
        COUNT(*) FILTER (WHERE ttc IS NULL OR cardinality(ttc) = 0)
            AS missing_ttc,
        ROUND(COUNT(*) FILTER (WHERE ttc IS NOT NULL AND cardinality(ttc) > 0)
              * 100.0 / COUNT(*), 1)
            AS coverage_pct,
        COUNT(*) FILTER (WHERE list_contains(map_keys(ttc), YEAR(plantstart) - 1))
            AS has_correct_yr,
        COUNT(*) FILTER (WHERE ttc IS NOT NULL AND cardinality(ttc) > 0
                           AND NOT list_contains(map_keys(ttc), YEAR(plantstart) - 1))
            AS has_ttc_wrong_yr,
        MIN(YEAR(plantstart) - 1) AS pred_yr_min,
        MAX(YEAR(plantstart) - 1) AS pred_yr_max
    FROM read_parquet('temp/tm.geoparquet')
    WHERE framework_key = 'terrafund-landscapes'
    GROUP BY 1
    ORDER BY missing_ttc DESC, short_name
) TO '/dev/stdout' (FORMAT CSV, HEADER);
