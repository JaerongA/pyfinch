-- By Jaerong
-- Create table for song data based on cluster table

CREATE TABLE IF NOT EXISTS song(
    id                   INTEGER PRIMARY KEY,
    birdID                   STRING  not null,
    taskName                 STRING  not null,
    taskSession              INT     not null,
    taskSessionDeafening     INT     not null,
    taskSessionPostDeafening INT     not null,
    dph                      INT     not null,
    block10days              INT     not null,
    sessionDate              DATE    not null,
    songNote                 STRING,
    motif                    STRING,
    introNotes               STRING,
    calls                    STRING,
    callSeqeunce             STRING,

    nbFilesUndir             INT,
    nbFilesDir               INT,

    nbBoutsUndir             INT,
    nbBoutsDir               INT,

    nbMotifsUndir            INT,
    nbMotifsDir              INT,

    meanIntroUndir           REAL,
    meanIntroDir             REAL,

    songCallPropUndir        REAL,
    songCallPropDir          REAL,

    motifDurationUndir       REAL,
    motifDurationDir         REAL,
    motifDurationCVUndir     REAL,
    motifDurationCVDir       REAL,
    UNIQUE (birdID, taskName, taskSession)
);

INSERT OR IGNORE INTO song (birdID, taskName, taskSession, taskSessionDeafening,
            taskSessionPostDeafening, dph, block10days, sessionDate,
            songNote, motif, introNotes, calls, callSeqeunce)
SELECT DISTINCT birdID, taskName, taskSession, taskSessionDeafening,
            taskSessionPostDeafening, dph, block10days, sessionDate,
            songNote, motif, introNotes, calls, callSeqeunce
FROM cluster;

