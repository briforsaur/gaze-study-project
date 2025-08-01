$n_participants = 30
$data_path = $Env:processed_data_path
$fig_path = $Env:trend_fig_path
for ( $i=1; $i -le $n_participants; $i++ )
{
    $participant = "P" + '{0:d2}' -f $i
    Write-Output "Plotting $participant"
    python .\scripts\conference_paper\plotting.py `
        $data_path `
        $participant `
        --fig_path $fig_path
}