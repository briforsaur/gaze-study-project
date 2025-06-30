$n_participants = 30
$data_path = "C:/Users/Owner/OneDrive - Dalhousie University/vision-intent-study/data-backup/processed_data/processed_data_f18.hdf5"
$fig_path = "C:/Users/Owner/OneDrive - Dalhousie University/vision-intent-study/results/figures/trendlines"
for ( $i=1; $i -le $n_participants; $i++ )
{
    $participant = "P" + '{0:d2}' -f $i
    Write-Output "Plotting $participant"
    python .\scripts\conference_paper\plotting.py `
        $data_path `
        $participant `
        --fig_path $fig_path
}