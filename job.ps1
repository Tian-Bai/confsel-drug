$datasets = @("3A4", "CB1", "DPP4", "HIVINT", "HIVPROT", "LOGD", "METAB", "NK1", "OX1", "OX2", "PGP", "PPB", "RAT_F", "TDI", "THROMBIN")
$sample = "0.1"

foreach ($dataset in $datasets) {
    for ($i = 1; $i -le 100; $i++) {
        python sheridan-vs-conformal.py $dataset $sample $i

        if ($i % 5 -eq 0) {
            Write-Output "$dataset $i done."
        }
    }
}