-- extracts the hansard statements from the 43rd parliament into json records
SELECT 
  row_to_json(record)
FROM
(
  SELECT
    hansards_statement.document_id,
    hansards_statement.sequence,
    hansards_statement.time,
    hansards_statement.content_en,
    hansards_statement.content_fr,
    hansards_statement.politician_id,
    hansards_statement.member_id,
    core_electedmember.riding_id,
    core_electedmember.party_id
  FROM 
    hansards_statement
  INNER JOIN 
    core_electedmember
  ON 
    hansards_statement.member_id = core_electedmember.id
  WHERE 
    hansards_statement.document_id
  IN
  (
    SELECT
      hansards_document.id
    FROM
      hansards_document
    WHERE
      hansards_document.session_id
    IN
    (
      '43-1',
      '43-2'
    )
  )
  ORDER BY hansards_statement.time
)
record;
