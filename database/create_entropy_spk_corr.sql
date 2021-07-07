-- Information about individual syllables per sessions

CREATE TABLE IF NOT EXISTS entropy_spk_corr
(
    syllableID               INTEGER PRIMARY KEY,
    clusterID                INTEGER NOT NULL,
    birdID                   STRING,
    taskName                 STRING,
    taskSession              STRING,
    note                     STRING,

    nbNoteUndir              INT,
    nbNoteDir                INT,

    premotorFRUndir          REAL,
    premotorFRDir            REAL,

    entropyUndir             REAL,  -- spectral entropy
    entropyDir               REAL,

    spectroTemporalEntropyUndir  REAL,  -- spectro-temporal entropy
    spectroTemporalEntropyDir   REAL,

    entropyVarUndir          REAL,
    entropyVarDir            REAL,

    spkCorrSigUndir          BOOL,   -- significance of correlation
    spkCorrSigDir            BOOL,

    spkCorrUndir             REAL,
    spkCorrDir               REAL,

    spkCorrRsquareUndir      REAL,
    spkCorrRsquareDir        REAL,

    shuffledSigPropUndir     INT,
    shuffledSigPropDir       INT,

    FOREIGN KEY (clusterID) REFERENCES song (id),
    UNIQUE (clusterID, note)
)

