-- DB for FF (fundamental frequency analysis)

CREATE TABLE IF NOT EXISTS ff_spk_corr(
            id                       INTEGER PRIMARY KEY,
            clusterID                INTEGER NOT NULL,
            birdID                   STRING,
            taskName                 STRING,
            taskSession              INT,
            note                     STRING NOT NULL,
            nbNoteUndir              INT,
            nbNoteDir                INT,
            ffMeanUndir              REAL,
            ffMeanDir                REAL,
            ffUndirCV                REAL,
            ffDirCV                  REAL,
            premotorFRUndir          REAL,
            premotorFRDir            REAL,
            spkCorrRUndir            REAL,   -- correlation between number of spikes and FF
            spkCorrRDir              REAL,   -- correlation between number of spikes and FF
            spkCorrPvalSigUndir      BOOL,   -- significance of correlation
            spkCorrPvalSigDir        BOOL,   -- significance of correlation
            polarityUndir            STRING,  -- positive vs. negative
            polarityDir              STRING,  -- positive vs. negative
            spkCorrRsquareUndir      REAL,
            spkCorrRsquareDir        REAL,
            shuffledSigPropUndir     REAL,
            shuffledSigPropDir       REAL,
            UNIQUE (clusterID, note)
            );
