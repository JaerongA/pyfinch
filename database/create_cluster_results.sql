-- Create cluster results database

CREATE TABLE IF NOT EXISTS cluster_results(
            clusterID     INTEGER NOT NULL UNIQUE,

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

            FOREIGN KEY(clusterID) REFERENCES cluster(id));
