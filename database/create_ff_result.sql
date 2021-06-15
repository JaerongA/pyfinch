-- DB for FF (fundamental frequency analysis)

CREATE TABLE IF NOT EXISTS ff_result(
            songID                   INTEGER,
            birdID                   STRING,
            taskName                 STRING  not null,
            taskSession              INT     not null,
            taskSessionDeafening     INT     not null,
            taskSessionPostDeafening INT     not null,
            block10days              INT     not null,
            note                     STRING,
            nbNoteUndir              INT,
            nbNoteDir                INT,
            ffMeanUndir              REAL,
            ffMeanDir                REAL,
            ffUndirCV                REAL,
            ffDirCV                  REAL
            );
