-- extracts politican names and ids
SELECT 
  row_to_json(record)
FROM
(
  SELECT
    id, name
  FROM 
    core_politician
)
record;
