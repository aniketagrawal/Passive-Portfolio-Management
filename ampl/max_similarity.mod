set ASSETS;  # Set of assets (e.g., equity names)
param q;     # Max number of assets in index fund
param rho{ASSETS, ASSETS};  # Asset similarities (correlation matrix)
param origindex{ASSETS};    # Weights in original index (to compute final positions)

var X{i in ASSETS, j in ASSETS} binary; # whether asset j is paired with asset i
var Y{i in ASSETS} binary;              # whether asset i is used in the index fund

maximize similarity: sum{i in ASSETS, j in ASSETS} rho[i,j]*X[i,j];  # Maximize similarity

subject to totassets: sum{i in ASSETS} Y[i] <= q;
subject to linking{i in ASSETS, j in ASSETS}: X[i,j] <= Y[i];
subject to oneassign{j in ASSETS}: sum{i in ASSETS} X[i,j] = 1;
