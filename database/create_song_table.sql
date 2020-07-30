-- By Jaerong
-- Create table for song data based on cluster table

CREATE TABLE IF NOT EXISTS song
(id INTEGER PRIMARY KEY AUTOINCREMENT,
birdID                   STRING  not null,
taskName                 STRING  not null,
taskSession              INT     not null,
taskSessionDeafening     INT     not null,
taskSessionPostDeafening INT     not null,
dph                      INT     not null,
block10days              INT     not null,
sessionDate              DATE    not null
);

INSERT INTO song
SELECT DISTINCT birdID, taskName, taskSession, taskSessionDeafening,
                taskSessionPostDeafening, dph, block10days, sessionDate
FROM cluster;


--CREATE VIEW song AS
--SELECT DISTINCT birdID, taskName, taskSession, taskSessionDeafening, taskSessionPostDeafening,
--           dph, block10days, sessionDate FROM cluster;




