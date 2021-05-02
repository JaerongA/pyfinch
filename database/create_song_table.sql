-- By Jaerong
-- Create table for song data based on cluster table
--
-- CREATE TABLE IF NOT EXISTS song2
-- (
-- birdID                   STRING  not null,
-- taskName                 STRING  not null,
-- taskSession              INT     not null,
-- taskSessionDeafening     INT     not null,
-- taskSessionPostDeafening INT     not null,
-- dph                      INT     not null,
-- block10days              INT     not null,
-- sessionDate              DATE    not null
-- );
--
-- INSERT INTO song2
-- SELECT DISTINCT birdID, taskName, taskSession, taskSessionDeafening,
--                 taskSessionPostDeafening, dph, block10days, sessionDate,
--                 songNote, motif, introNotes, calls, callSeqeunce
-- FROM cluster;
--
--
-- CREATE TABLE song2 AS
--     SELECT DISTINCT birdID, taskName, taskSession, taskSessionDeafening,
--                 taskSessionPostDeafening, dph, block10days, sessionDate,
--                 songNote, motif, introNotes, calls, callSeqeunce
--     FROM cluster;

CREATE TABLE unit_profile(
            cluster_id     INTEGER,
            birdID         STRING  not null,
            taskName       STRING  not null,
            taskSession    INTEGER not null,
            site           STRING  not null,
            channel        STRING  not null,
            unit           STRING,
            region         STRING,
            isiPeakLatency REAL,
            withinRefProp  REAL,
            FOREIGN KEY(cluster_id) REFERENCES cluster(id))



