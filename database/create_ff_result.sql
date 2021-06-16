-- DB for FF (fundamental frequency analysis)

CREATE TABLE IF NOT EXISTS ff_result(
            id                       INTEGER PRIMARY KEY,
            songID                   INTEGER NOT NULL,
            birdID                   STRING,
            taskName                 STRING,
            taskSession              INT,
            taskSessionDeafening     INT,
            taskSessionPostDeafening INT,
            block10days              INT,
            note                     STRING NOT NULL,
            nbNoteUndir              INT,
            nbNoteDir                INT,
            ffMeanUndir              REAL,
            ffMeanDir                REAL,
            ffUndirCV                REAL,
            ffDirCV                  REAL,
            UNIQUE (songID, note)
            );
