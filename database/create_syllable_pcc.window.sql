-- Store results from raster_syllable_window
-- Calculate PCC across shifting time windows

CREATE TABLE IF NOT EXISTS syllable_pcc_window
(
    syllableID               INTEGER PRIMARY KEY,
    clusterID                INTEGER NOT NULL,
    birdID                   STRING,
    taskName                 STRING,
    taskSession              INT,
    taskSessionDeafening     INT,
    taskSessionPostDeafening INT,
    dph                      INT,
    block10days              INT,
    note                     STRING,

    nbNoteUndir              INT,
    nbNoteDir                INT,

    pccUndirPre              REAL,
    pccDirPre                REAL,

    pccUndirSyllable         REAL,
    pccDirSyllable           REAL,

    pccUndirPost             REAL,
    pccDirPost               REAL,

    FOREIGN KEY (clusterID) REFERENCES cluster (id),
    UNIQUE (clusterID, note)
)

