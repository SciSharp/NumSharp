#Requires -Version 7.0
<#
.SYNOPSIS
Verify the four Gist coverage maps against final managed/live TRX results on both frameworks.
.DESCRIPTION
No build, download or test execution occurs. Missing/ambiguous references, absent DataRows,
non-passing outcomes, changed source pins and unexpected test totals are hard errors.
Exclusions and unreproduced claims remain records, never converted into passes because a
related synthetic test passed. The historical verification.json is never overwritten.
.EXAMPLE
pwsh -File examples/gist/verify-demonstrations.ps1 -PythonExecutable C:/Users/ELI/.claude/python/python.exe
#>
[CmdletBinding()]
param(
    [string]$RepositoryRoot = (Join-Path $PSScriptRoot '../..'),
    [string]$LogDirectory = 'outputs/gist-demonstrations-2026-09-09',
    [string]$CoverageDirectory = 'examples/gist',
    [string]$OutputPath = 'examples/gist/demonstration-verification.json',
    [int]$ExpectedManagedPerFramework = 88,
    [int]$ExpectedLivePerFramework = 55,
    [string]$PythonExecutable = '',
    [switch]$SelfTest
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repo = [IO.Path]::GetFullPath($RepositoryRoot).TrimEnd([IO.Path]::DirectorySeparatorChar, [IO.Path]::AltDirectorySeparatorChar)
$filePins = @{}
$utf8 = [Text.UTF8Encoding]::new($false)

function Resolve-RepoPath([string]$Path) {
    $full = if ([IO.Path]::IsPathRooted($Path)) { [IO.Path]::GetFullPath($Path) } else { [IO.Path]::GetFullPath((Join-Path $repo $Path)) }
    if (-not $full.StartsWith($repo + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Path must be a file beneath the repository: $Path"
    }
    return $full
}
function Relative-Path([string]$Path) { [IO.Path]::GetRelativePath($repo, (Resolve-RepoPath $Path)).Replace('\', '/') }
function Read-Json([string]$Path) { Get-Content -LiteralPath (Resolve-RepoPath $Path) -Raw | ConvertFrom-Json -AsHashtable }
function Pin-File([string]$Path, [string]$Expected = '') {
    $full = Resolve-RepoPath $Path
    if (-not (Test-Path -LiteralPath $full -PathType Leaf)) { throw "Required file is missing: $Path" }
    $relative = Relative-Path $full
    $hash = (Get-FileHash -LiteralPath $full -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($Expected -and $hash -ne $Expected.ToLowerInvariant()) { throw "SHA256 mismatch: $relative" }
    if ($filePins.ContainsKey($relative) -and $filePins[$relative].sha256 -ne $hash) { throw "File changed during verification: $relative" }
    $pin = [ordered]@{ path = $relative; sha256 = $hash; bytes = (Get-Item -LiteralPath $full).Length }
    $filePins[$relative] = $pin
    return $pin
}
function Sha-Text([string]$Text) { [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($utf8.GetBytes($Text))).ToLowerInvariant() }
function Get-Value([Collections.IDictionary]$Map, [string]$Key, $Default = $null) {
    if ($Map.Contains($Key)) { return $Map[$Key] }
    return $Default
}

$outputFull = Resolve-RepoPath $OutputPath
if ([IO.Path]::GetFileName($outputFull) -eq 'verification.json') { throw 'Refusing to overwrite historical verification.json.' }
$sourceInventoryPath = 'examples/gist/sources.json'
$inventoryPin = Pin-File $sourceInventoryPath
$inventory = Read-Json $sourceInventoryPath
$sourceRecords = [Collections.Generic.List[object]]::new()
$sourceByRank = @{}
$implementationPins = [Collections.Generic.List[object]]::new()
foreach ($gist in $inventory.gists) {
    $facts = $gist.facts
    $rank = [int]$facts.saved_ranking_row.rank
    if ($sourceByRank.ContainsKey($rank)) { throw "Duplicate source rank $rank" }
    $sourceByRank[$rank] = [Collections.Generic.List[object]]::new()
    [void](Pin-File $facts.files_manifest.path $facts.files_manifest.sha256)
    $apiPin = Pin-File $facts.pinned_api_evidence.path $facts.pinned_api_evidence.sha256
    $api = Read-Json $facts.pinned_api_evidence.path
    foreach ($file in $facts.source_files) {
        $manifest = $file.manifest_entry
        $pin = Pin-File $file.verification.local_path $manifest.sha256
        $record = [ordered]@{ rank = $rank; filename = $manifest.path; revision = $manifest.revision;
            source_url = $manifest.source_url; cache = $pin; source_kind = 'pinned_source_file' }
        $sourceByRank[$rank].Add($record); $sourceRecords.Add($record)
    }
    foreach ($auxiliary in $facts.pinned_api_evidence.auxiliary_files) {
        $name = $auxiliary.filename
        if (-not $api.files.Contains($name) -or $api.files[$name].truncated) { throw "Missing/truncated auxiliary source: rank $rank $name" }
        $hash = Sha-Text $api.files[$name].content
        if ($hash -ne $auxiliary.content_utf8_sha256) { throw "Auxiliary source SHA256 mismatch: rank $rank $name" }
        $record = [ordered]@{ rank = $rank; filename = $name; revision = $facts.saved_ranking_row.revision;
            source_url = $auxiliary.raw_url; source_kind = 'pinned_api_embedded_auxiliary'; sha256 = $hash;
            bytes = $utf8.GetByteCount($api.files[$name].content); containing_api_cache = $apiPin }
        $sourceByRank[$rank].Add($record); $sourceRecords.Add($record)
    }
    $implementationPins.Add((Pin-File $gist.port.file))
}
if ($sourceByRank.Count -ne 10) { throw 'Expected exactly ten ranked gists in the source inventory.' }

# Parse the small, conventional MSTest source surface. Unknown dynamic data is rejected rather
# than guessing how many rows should have run. This catches a missing DataRow even if the TRX's
# own counters consistently describe the incomplete discovery.
$sourceTests = @{}
$testPins = [Collections.Generic.List[object]]::new()
foreach ($directory in @('test/NumSharp.Tests/Examples', 'test/NumSharp.Tests.Interop')) {
    foreach ($file in Get-ChildItem -LiteralPath (Resolve-RepoPath $directory) -Filter 'Gist*.cs' -File) {
        $testPins.Add((Pin-File $file.FullName))
        $content = Get-Content -LiteralPath $file.FullName -Raw
        if ($content -notmatch '\[TestClass') { continue }
        $namespace = [regex]::Match($content, '(?m)^\s*namespace\s+([\w.]+)\s*[;{]').Groups[1].Value
        $class = [regex]::Match($content, '(?m)^\s*public\s+(?:sealed\s+|partial\s+)?class\s+(\w+)').Groups[1].Value
        if (-not $namespace -or -not $class) { throw "Cannot resolve test namespace/class in $($file.Name)" }
        $matches = [regex]::Matches($content, '(?ms)(?<attributes>(?:\s*\[[^\]]+\]\s*)+)public\s+(?:async\s+)?(?:void|Task)\s+(?<method>\w+)\s*\(')
        foreach ($match in $matches) {
            $attributes = $match.Groups['attributes'].Value
            if ($attributes -notmatch '\[(?:Data)?TestMethod\b') { continue }
            if ($attributes -match '\bDynamicData\b') { throw "DynamicData needs an explicit verifier adapter: $class" }
            $name = "$namespace.$class.$($match.Groups['method'].Value)"
            if ($sourceTests.ContainsKey($name)) { throw "Ambiguous source test: $name" }
            $rows = [regex]::Matches($attributes, '\[DataRow\s*\(').Count
            if ($attributes -match '\[DataTestMethod\b' -and $rows -eq 0) { throw "DataTestMethod has no discoverable rows: $name" }
            $sourceTests[$name] = [ordered]@{ method = $name; file = Relative-Path $file.FullName;
                declared_data_rows = $rows; expected_executions = [Math]::Max(1, $rows);
                suite = $(if ($directory -like '*Interop') { 'live' } else { 'unit' }) }
        }
    }
}
foreach ($path in @('examples/gist/run.cs','examples/gist/NumSharp.GistExamples.csproj',
    'test/NumSharp.Tests/NumSharp.Tests.csproj','test/NumSharp.Tests.Interop/NumSharp.Tests.Interop.csproj',
    'test/NumSharp.Tests.Interop/InteropTestBase.cs','test/NumSharp.Tests.Interop/PythonSession.cs',
    'test/NumSharp.Tests.Interop/Pythonic.cs','examples/gist/demo-output.txt',
    'examples/gist/README.md','examples/gist/VERIFICATION.md','examples/gist/DEMONSTRATION_COVERAGE.md',
    'outputs/gist-demonstrations-2026-09-09/file-runner-final.txt',
    'outputs/gist-demonstrations-2026-09-09/file-runner-final.json')) {
    if (Test-Path -LiteralPath (Resolve-RepoPath $path) -PathType Leaf) { [void](Pin-File $path) }
}

$runs = [Collections.Generic.List[object]]::new()
$executions = [Collections.Generic.List[object]]::new()
$runMethods = @{}
foreach ($framework in @('net8.0','net10.0')) {
    foreach ($suite in @('unit','live')) {
        $token = $framework.Replace('.0','')
        $path = Join-Path $LogDirectory "source-demo-$suite-$token-final-v3.trx"
        $logPin = Pin-File $path
        [xml]$xml = Get-Content -LiteralPath (Resolve-RepoPath $path) -Raw
        $definitions = @($xml.SelectNodes("/*[local-name()='TestRun']/*[local-name()='TestDefinitions']/*[local-name()='UnitTest']"))
        $results = @($xml.SelectNodes("/*[local-name()='TestRun']/*[local-name()='Results']/*[local-name()='UnitTestResult']"))
        $summary = $xml.SelectSingleNode("/*[local-name()='TestRun']/*[local-name()='ResultSummary']")
        $counters = $summary.SelectSingleNode("*[local-name()='Counters']")
        $expectedCount = if ($suite -eq 'unit') { $ExpectedManagedPerFramework } else { $ExpectedLivePerFramework }
        if ($summary.outcome -ne 'Completed' -or $results.Count -ne $expectedCount -or
            [int]$counters.total -ne $expectedCount -or [int]$counters.executed -ne $expectedCount -or
            [int]$counters.passed -ne $expectedCount -or $definitions.Count -ne $expectedCount) {
            throw "Incomplete/unexpected run $path; expected $expectedCount passing definitions/results."
        }
        $byId = @{}; $seenResults = @{}; $methodRows = @{}
        foreach ($definition in $definitions) {
            $id = [string]$definition.id
            if ($byId.ContainsKey($id)) { throw "Duplicate TRX definition $id in $path" }
            $methodNode = $definition.SelectSingleNode("*[local-name()='TestMethod']")
            $name = "$($methodNode.className).$($methodNode.name)"
            if ($methodNode.className.Split('.')[-1] -notlike 'Gist*') { throw "Unrelated test in Gist-only run: $name" }
            if (-not $sourceTests.ContainsKey($name) -or $sourceTests[$name].suite -ne $suite) { throw "TRX definition has no unique current test source: $name" }
            if ($methodNode.codeBase.Replace('\','/') -notmatch "/$([regex]::Escape($framework))/") { throw "Framework/path mismatch for $name in $path" }
            $byId[$id] = @{ node=$definition; method=$name; codeBase=[string]$methodNode.codeBase }
        }
        foreach ($result in $results) {
            $id = [string]$result.testId
            if (-not $byId.ContainsKey($id) -or $seenResults.ContainsKey($id)) { throw "Missing/duplicate TRX result definition $id in $path" }
            $seenResults[$id] = $true
            if ($result.outcome -ne 'Passed') { throw "Failed/skipped test in $path : $($result.testName) ($($result.outcome))" }
            $definition = $byId[$id]
            $declaredExecution = $definition.node.SelectSingleNode("*[local-name()='Execution']").id
            if ($declaredExecution -ne $result.executionId) { throw "Execution-id mismatch: $id" }
            $key = "$framework/$suite/$($result.executionId)"
            $row = [ordered]@{ key=$key; framework=$framework; suite=$suite; fully_qualified_method=$definition.method;
                display_name=[string]$result.testName; outcome=[string]$result.outcome; test_id=$id;
                execution_id=[string]$result.executionId; log=$logPin.path; duration=[string]$result.duration;
                started_at=[string]$result.startTime; finished_at=[string]$result.endTime; computer=[string]$result.computerName }
            $executions.Add($row)
            if (-not $methodRows.ContainsKey($definition.method)) { $methodRows[$definition.method] = [Collections.Generic.List[object]]::new() }
            $methodRows[$definition.method].Add($row)
        }
        foreach ($name in $sourceTests.Keys) {
            if ($sourceTests[$name].suite -ne $suite) { continue }
            if (-not $methodRows.ContainsKey($name) -or $methodRows[$name].Count -ne $sourceTests[$name].expected_executions) {
                throw "Missing method/DataRows in $path : $name; source requires $($sourceTests[$name].expected_executions) executions."
            }
            if (@($methodRows[$name] | ForEach-Object { $_.display_name } | Sort-Object -Unique).Count -ne $methodRows[$name].Count) {
                throw "Duplicate DataRow display name in $path : $name"
            }
        }
        $runMethods["$framework/$suite"] = $methodRows
        $assemblies = @($byId.Values.codeBase | Sort-Object -Unique | ForEach-Object { Pin-File $_ })
        $times = $xml.SelectSingleNode("/*[local-name()='TestRun']/*[local-name()='Times']")
        $runs.Add([ordered]@{ framework=$framework; suite=$suite; expected_count=$expectedCount; passed=$results.Count;
            methods=$methodRows.Count; log=$logPin; started_at=[string]$times.start; finished_at=[string]$times.finish;
            current_test_assembly_pins=$assemblies })
    }
}

# Explicit field adapters: an unknown future test-reference field fails instead of being ignored.
$referenceFields = @('tests','test_methods','managed_test','live_test','named_entrypoint_test','additional_managed_test',
    'additional_existing_tests','existing_managed_tests','existing_live_test','existing_edge_test','existing_tie_test')
function Extract-References([Collections.IDictionary]$Case) {
    $references = [Collections.Generic.List[object]]::new()
    foreach ($key in $Case.Keys) {
        if ($key -match 'test' -and $key -notin $referenceFields) { throw "Unknown test-reference field '$key' in a coverage case." }
        if ($key -notin $referenceFields) { continue }
        foreach ($value in @($Case[$key])) {
            if ($null -eq $value) { continue }
            if ($value -is [string]) { $references.Add(@{ reference=$value; field=$key; declared_file=$null }) }
            elseif ($value -is [Collections.IDictionary] -and $value.Contains('method')) {
                $references.Add(@{ reference=$value.method; field=$key; declared_file=(Get-Value $value 'file') })
            } else { throw "Unsupported test reference in field $key" }
        }
    }
    return $references.ToArray()
}
function Classify-Case([Collections.IDictionary]$Case, [bool]$ExplicitExclusion) {
    $status = [string](Get-Value $Case 'status' '')
    $kind = [string](Get-Value $Case 'kind' '')
    if ($ExplicitExclusion -or $kind -like 'excluded_*' -or $status -like 'outside_*' -or $status -eq 'non_numerical_presentation') { return 'excluded' }
    if ($status -in @('numerical_paths_asserted_input_not_reproduced','algebra_illustrated_external_claim_not_validated')) { return 'related_assertions_only' }
    if ($status -like 'not_*' -or $status -in @('documented_not_an_executable_example','recorded_not_a_numerical_demonstration')) { return 'not_reproduced_or_not_executable' }
    if (-not $status -or $status -like 'asserted*' -or $status -eq 'illustrated_with_asserted_counterexample') { return 'verified_assertions' }
    throw "Unknown coverage status '$status'; define its meaning before claiming verification."
}

$caseRecords = [Collections.Generic.List[object]]::new()
$seenCases = @{}
$coveragePins = [Collections.Generic.List[object]]::new()
foreach ($mapName in @('learning','metrics','signal','tensor')) {
    $mapPath = Relative-Path (Join-Path $CoverageDirectory "coverage-$mapName.json")
    $coveragePins.Add((Pin-File $mapPath))
    $map = Read-Json $mapPath
    $entries = [Collections.Generic.List[object]]::new()
    if ($map.Contains('cases')) {
        for ($i=0; $i -lt $map.cases.Count; $i++) { $entries.Add(@{ value=$map.cases[$i]; rank=$map.cases[$i].gist_rank; pointer="/cases/$i"; excluded=$false; source=$null }) }
    } else {
        for ($i=0; $i -lt $map.sources.Count; $i++) {
            $source = $map.sources[$i]
            foreach ($group in @('cases','exclusions')) {
                if (-not $source.Contains($group)) { continue }
                for ($j=0; $j -lt $source[$group].Count; $j++) {
                    $entries.Add(@{ value=$source[$group][$j]; rank=$source.rank; pointer="/sources/$i/$group/$j"; excluded=($group -eq 'exclusions'); source=$source })
                }
            }
        }
    }
    foreach ($entry in $entries) {
        $case = $entry.value
        $id = [string](Get-Value $case 'case_id' (Get-Value $case 'id'))
        if (-not $id -or $seenCases.ContainsKey($id)) { throw "Missing/duplicate coverage case ID: $id" }
        $seenCases[$id] = $true
        $rank = [int]$entry.rank
        if (-not $sourceByRank.ContainsKey($rank)) { throw "Unknown gist rank $rank for $id" }
        $classification = Classify-Case $case $entry.excluded
        $references = @(Extract-References $case)
        if ($classification -eq 'verified_assertions' -and $references.Count -eq 0) { throw "Asserted coverage case has no test: $id" }
        $resolved = [Collections.Generic.List[object]]::new()
        foreach ($reference in $references) {
            $matches = @($sourceTests.Keys | Where-Object { $_ -eq $reference.reference -or $_.EndsWith('.' + $reference.reference, [StringComparison]::Ordinal) })
            if ($matches.Count -ne 1) { throw "Missing/ambiguous test '$($reference.reference)' in $id; found $($matches.Count)." }
            $name = $matches[0]; $sourceTest = $sourceTests[$name]
            if ($reference.declared_file -and (Relative-Path $reference.declared_file) -ne $sourceTest.file) { throw "Wrong declared test file for $name" }
            $bindings = [Collections.Generic.List[object]]::new()
            foreach ($framework in @('net8.0','net10.0')) {
                $rows = $runMethods["$framework/$($sourceTest.suite)"][$name]
                $selected = @($rows)
                if ($case.Contains('data_row')) {
                    $shortMethod = $name.Split('.')[-1]
                    $display = "$shortMethod ($($case.data_row))"
                    $selected = @($rows | Where-Object { $_.display_name -eq $display })
                    if ($selected.Count -ne 1) { throw "Missing/ambiguous required DataRow $display for $id on $framework" }
                }
                $bindings.Add([ordered]@{ framework=$framework; suite=$sourceTest.suite;
                    all_method_executions=@($rows | ForEach-Object { $_.key }); all_data_rows_passed=$true;
                    case_selected_executions=@($selected | ForEach-Object { $_.key }) })
            }
            $resolved.Add([ordered]@{ reference=$reference.reference; field=$reference.field; fully_qualified_method=$name;
                current_source_file=$sourceTest.file; declared_data_rows=$sourceTest.declared_data_rows; bindings=$bindings.ToArray() })
        }
        $filename = [string](Get-Value $case 'source_file' (Get-Value $case 'source_path'))
        if (-not $filename -and $entry.source -and $entry.source.Contains('filename')) { $filename = $entry.source.filename }
        $associatedSources = @(if ($filename) { $sourceByRank[$rank] | Where-Object { $_.filename -eq $filename } } else { $sourceByRank[$rank] })
        if ($filename -and $associatedSources.Count -ne 1) { throw "Unresolved source '$filename' for $id" }
        if ($case.Contains('source_sha256')) {
            $sourceHash = if ($associatedSources[0].Contains('cache')) { $associatedSources[0].cache.sha256 } else { $associatedSources[0].sha256 }
            if ($case.source_sha256 -ne $sourceHash) { throw "Coverage/source hash mismatch for $id" }
        }
        $caseRecords.Add([ordered]@{ case_id=$id; rank=$rank; classification=$classification;
            coverage_map=$mapPath; json_pointer=$entry.pointer; original_claim_status=(Get-Value $case 'status');
            source_links=@($associatedSources | ForEach-Object { $_.source_url });
            verified_test_reference_count=$resolved.Count; references=$resolved.ToArray(); original_record=$case })
    }
}

$demoCases = @($caseRecords | Where-Object { (Get-Value $_.original_record 'kind') -eq 'ported_demo_output_regression' })
if ($demoCases.Count -ne 10 -or @($demoCases | ForEach-Object { $_['rank'] } | Sort-Object -Unique).Count -ne 10) { throw 'All ten actual Demo output rows must be represented.' }
$classificationCounts = [ordered]@{}
foreach ($group in $caseRecords | Group-Object -Property { $_['classification'] }) { $classificationCounts[$group.Name] = $group.Count }
$environment = [ordered]@{ observed_at_utc=[DateTime]::UtcNow.ToString('o'); powershell=$PSVersionTable.PSVersion.ToString();
    os=[Runtime.InteropServices.RuntimeInformation]::OSDescription; architecture=[Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString();
    dotnet_sdk=((& dotnet --version) -join "`n").Trim(); installed_dotnet_runtimes=@(& dotnet --list-runtimes);
    test_computers=@($executions | ForEach-Object { $_['computer'] } | Sort-Object -Unique); python_probe=$null;
    observed_git_head=((& git -C $repo rev-parse HEAD) -join "`n").Trim();
    limitation='These are current verification-process observations; TRX independently records test host names/times. Current file hashes do not by themselves establish historical build lineage.' }
if ($PythonExecutable) {
    $pythonDetails = & $PythonExecutable -c 'import sys,json,numpy,scipy; print(json.dumps({"executable":sys.executable,"python":sys.version,"numpy":numpy.__version__,"scipy":scipy.__version__}))'
    if ($LASTEXITCODE -ne 0) { throw 'Explicit Python environment probe failed.' }
    $environment.python_probe = $pythonDetails | ConvertFrom-Json -AsHashtable
}
$rejectionChecks = [Collections.Generic.List[object]]::new()
if ($SelfTest) {
    # Materialize fresh bounded fixtures; never mutate or delete the real maps/logs. Copies are
    # deliberately retained with the report's rejection evidence for independent inspection.
    $fixtureRoot = Resolve-RepoPath (Join-Path $LogDirectory ('verifier-self-test-' + [Guid]::NewGuid().ToString('N')))
    foreach ($scenario in @('missing_reference','failed_data_row','skipped_data_row','wrong_class','missing_data_row','duplicate_data_row')) {
        $fixture = Join-Path $fixtureRoot $scenario
        $maps = Join-Path $fixture 'maps'; $logs = Join-Path $fixture 'logs'
        [void][IO.Directory]::CreateDirectory($maps); [void][IO.Directory]::CreateDirectory($logs)
        foreach ($mapName in @('learning','metrics','signal','tensor')) {
            Copy-Item -LiteralPath (Resolve-RepoPath (Join-Path $CoverageDirectory "coverage-$mapName.json")) -Destination $maps
        }
        foreach ($frameworkToken in @('net8','net10')) {
            foreach ($suite in @('unit','live')) {
                Copy-Item -LiteralPath (Resolve-RepoPath (Join-Path $LogDirectory "source-demo-$suite-$frameworkToken-final-v3.trx")) -Destination $logs
            }
        }
        if ($scenario -eq 'missing_reference') {
            $badMapPath = Join-Path $maps 'coverage-learning.json'
            $badMap = Get-Content -LiteralPath $badMapPath -Raw | ConvertFrom-Json -AsHashtable
            $badMap.cases[0].tests[0] = 'GistMissingClass.ThisMethodDoesNotExist'
            [IO.File]::WriteAllText($badMapPath, ($badMap | ConvertTo-Json -Depth 60), $utf8)
        } else {
            $badLogPath = Join-Path $logs 'source-demo-unit-net10-final-v3.trx'
            [xml]$badLog = Get-Content -LiteralPath $badLogPath -Raw
            $badRows = @($badLog.TestRun.Results.UnitTestResult | Where-Object { $_.testName -like 'EveryPortDemo_ReproducesAllItsPublishedMeasurements (*)' })
            if ($badRows.Count -ne 10) { throw 'Negative fixture setup requires the ten known Demo DataRows.' }
            switch ($scenario) {
                'failed_data_row' { $badRows[0].SetAttribute('outcome','Failed') }
                'skipped_data_row' { $badRows[0].SetAttribute('outcome','NotExecuted') }
                'wrong_class' {
                    $badDefinition = @($badLog.TestRun.TestDefinitions.UnitTest | Where-Object { $_.id -eq $badRows[0].testId })[0]
                    $badDefinition.TestMethod.SetAttribute('className','NumSharp.UnitTest.RandomLogisticTests')
                }
                'missing_data_row' { [void]$badRows[0].ParentNode.RemoveChild($badRows[0]) }
                'duplicate_data_row' { $badRows[1].SetAttribute('testName',[string]$badRows[0].testName) }
            }
            $badLog.Save($badLogPath)
        }
        $rejection = $null
        try {
            & $PSCommandPath -RepositoryRoot $repo -LogDirectory $logs -CoverageDirectory $maps `
                -OutputPath (Join-Path $fixture 'unexpected-success.json') `
                -ExpectedManagedPerFramework $ExpectedManagedPerFramework -ExpectedLivePerFramework $ExpectedLivePerFramework
        } catch { $rejection = $_.Exception.Message }
        if (-not $rejection) { throw "Verifier incorrectly accepted negative fixture: $scenario" }
        $rejectionChecks.Add([ordered]@{ scenario=$scenario; rejected=$true; reason=$rejection;
            fixture_directory=Relative-Path $fixture })
    }
}
$scriptPin = Pin-File $PSCommandPath
# Recheck everything just before committing the generated report; concurrent edits are failures.
foreach ($path in @($filePins.Keys)) { [void](Pin-File $path $filePins[$path].sha256) }
$report = [ordered]@{ schema_version=1; verified_at_utc=[DateTime]::UtcNow.ToString('o');
    verification_status='passed'; scope='Original demonstrations and explicitly adapted numerical cores; no framework/application completeness claim.';
    filter='FullyQualifiedName~.Gist'; expected_managed_per_framework=$ExpectedManagedPerFramework;
    expected_live_per_framework=$ExpectedLivePerFramework; total_test_executions=$executions.Count;
    failed_or_skipped_executions=0; source_inventory=$inventoryPin; verifier=$scriptPin; environment=$environment;
    coverage_case_count=$caseRecords.Count; classification_counts=$classificationCounts;
    verified_assertions_meaning='Referenced assertions passed on both frameworks. This does not turn an excluded/unreproduced historical claim or synthetic fixture into a reproduced original application.';
    all_ten_actual_demo_outputs_verified=$true; demo_cases=@($demoCases | ForEach-Object { $_['case_id'] });
    coverage_maps=$coveragePins.ToArray(); source_files=$sourceRecords.ToArray();
    current_implementation_pins=$implementationPins.ToArray(); current_test_source_pins=$testPins.ToArray();
    runs=$runs.ToArray(); executions=$executions.ToArray(); cases=$caseRecords.ToArray();
    verifier_rejection_checks=$rejectionChecks.ToArray();
    all_file_pins=@($filePins.Keys | Sort-Object | ForEach-Object { $filePins[$_] }) }
$parentDirectory = Split-Path -Parent $outputFull
if (-not (Test-Path -LiteralPath $parentDirectory -PathType Container)) { throw "Output directory must already exist: $parentDirectory" }
[IO.File]::WriteAllText($outputFull, ($report | ConvertTo-Json -Depth 60) + "`n", $utf8)
Write-Host "Verified $($executions.Count) passing executions, $($caseRecords.Count) coverage records, ten actual Demo outputs."
Write-Host "Exclusions and unreproduced claims are retained separately. Report: $(Relative-Path $outputFull)"
