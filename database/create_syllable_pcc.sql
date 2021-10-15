-- Store results from raster_syllable
CREATE TABLE IF NOT EXISTS syllable_pcc
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

    frUndir                  REAL,
    frDir                    REAL,

    pccUndir                 REAL,
    pccDir                   REAL,

    pccUndirSig              BOOLEAN,
    pccDirSig                BOOLEAN,

    corrContext              REAL,

    psdSimilarity            REAL,

    entropyUndir             REAL,
    entropyDir               REAL,

    spectroTempEntropyUndir  REAL,
    spectroTempEntropyDir    REAL,

    entropyVarUndir          REAL,
    entropyVarDir            REAL,

    FOREIGN KEY (clusterID) REFERENCES cluster (id),
    UNIQUE (clusterID, note)
)

