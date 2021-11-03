-- Create pcc database
-- Take values from cluster db

-- DROP TABLE IF EXISTS pcc;

CREATE TABLE IF NOT EXISTS pcc(
            clusterID     INTEGER NOT NULL UNIQUE,
            birdID                   STRING,
            taskName                 STRING,
            taskSession              INT,
            taskSessionDeafening     INT,
            taskSessionPostDeafening INT,
            dph                      INT,
            block10days              INT,

            baselineFR               REAL,
            motifFRUndir             REAL,
            motifFRDir               REAL,

            pccUndir                 REAL,
            pccDir                   REAL,
            corrRContext             REAL,

            pccUndirSig              REAL,
            pccDirSig                REAL,

            cvSpkCountUndir          REAL,
            cvSpkCountDir            REAL,

            fanoSpkCountUndir        REAL,
            fanoSpkCountDir          REAL,

            sparsenessUndir          REAL,
            sparsenessDir            REAL,

            FOREIGN KEY(clusterID) REFERENCES cluster(id));

INSERT OR IGNORE INTO pcc
    (clusterID, birdID, taskName, taskSession, taskSessionDeafening, taskSessionPostDeafening, dph, block10days)
SELECT id, birdID, taskName, taskSession, taskSessionDeafening, taskSessionPostDeafening, dph, block10days FROM main.cluster
WHERE analysisOK is TRUE