#Requires -Version 7.0
<#
.SYNOPSIS
Resolve Karpathy coverage claims against passing managed/live TRX on .NET 8 and .NET 10.
.DESCRIPTION
Read-only verification except for the generated report and optional retained negative fixtures.
No builds, test runs, downloads or original application execution. Test/DataRow totals derive
from the current test source and must reconcile with every final TRX definition and result.
.EXAMPLE
pwsh -File examples/gist/karpathy/verify.ps1 -PythonExecutable C:/Users/ELI/.claude/python/python.exe -SelfTest
#>
[CmdletBinding()]
param(
    [string]$RepositoryRoot = (Join-Path $PSScriptRoot '../../..'),
    [string]$LogDirectory = 'outputs/karpathy-parity-2026-09-09',
    [string]$AcquisitionDirectory = 'outputs/karpathy-gists-2026-09-09',
    [string]$FinalLogSuffix = 'final',
    [string]$CoverageDirectory = 'examples/gist/karpathy',
    [string]$OutputPath = 'examples/gist/karpathy/parity-verification.json',
    [string]$PythonExecutable = '',
    [switch]$SelfTest
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repo = [IO.Path]::GetFullPath($RepositoryRoot).TrimEnd([IO.Path]::DirectorySeparatorChar, [IO.Path]::AltDirectorySeparatorChar)
$utf8 = [Text.UTF8Encoding]::new($false)
$pins = @{}
function Full([string]$Path) {
    $result = [IO.Path]::GetFullPath($(if ([IO.Path]::IsPathRooted($Path)) { $Path } else { Join-Path $repo $Path }))
    if (-not $result.StartsWith($repo + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) { throw "Not a repository file: $Path" }
    return $result
}
function Rel([string]$Path) { [IO.Path]::GetRelativePath($repo,(Full $Path)).Replace('\','/') }
function Json([string]$Path) { Get-Content -LiteralPath (Full $Path) -Raw | ConvertFrom-Json -AsHashtable }
function Value([Collections.IDictionary]$Object, [string]$Key, $Default = $null) { if ($Object.Contains($Key)) { return $Object[$Key] }; return $Default }
function Pin([string]$Path, [string]$Expected = '') {
    $full = Full $Path
    if (-not (Test-Path -LiteralPath $full -PathType Leaf)) { throw "Required file missing: $Path" }
    $relative = Rel $full; $hash = (Get-FileHash -LiteralPath $full -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($Expected -and $hash -ne $Expected.ToLowerInvariant()) { throw "SHA256 mismatch: $relative" }
    if ($pins.ContainsKey($relative) -and $pins[$relative].sha256 -ne $hash) { throw "File changed during verification: $relative" }
    $record = [ordered]@{ path=$relative; sha256=$hash; bytes=(Get-Item -LiteralPath $full).Length }
    $pins[$relative]=$record; return $record
}
$outputFull = Full $OutputPath
if ($outputFull -in @((Full 'examples/gist/verification.json'),(Full 'examples/gist/demonstration-verification.json'),(Full 'examples/gist/karpathy/verification.json'))) {
    throw 'Historical verification records must not be overwritten.'
}
$inventoryPin = Pin 'examples/gist/karpathy/sources.json'
$inventory = Json $inventoryPin.path
[void](Pin $inventory.selection.catalog)
$sources = [Collections.Generic.List[object]]::new()
$originalSourceMap = @{}
foreach ($source in $inventory.sources) {
    if (-not $source.raw_url.Contains('/raw/' + $source.raw_file_revision + '/')) { throw "Raw-file revision/URL mismatch: $($source.key)" }
    $body = Pin $source.path $source.sha256; $receipt = Pin $source.receipt $source.receipt_sha256
    [void](Pin $source.port_file)
    $sources.Add([ordered]@{ original_record=$source; body=$body; receipt=$receipt;
        revision_claim=$(if ($null -eq $source.gist_revision) { 'immutable_raw_file_only; gist_HEAD_not_verified' } else { 'raw_file_and_separately_observed_gist_revision' }) })
    $originalSourceMap[$source.key]=@{path=$body.path;sha256=$body.sha256;source_url=$source.raw_url;inventory=$inventoryPin.path}
}
foreach($existing in $inventory.existing_ports) { [void](Pin $existing.file) }
# The two earlier Karpathy sources stay in the original top-ten inventory; never invent new pins.
$earlierInventoryPin=Pin 'examples/gist/sources.json'
$earlierInventory=Json $earlierInventoryPin.path
foreach($identity in @(@{key='nes';id='77fbb6a8dac5395f1b73e7a89300318d'},@{key='walk';id='00103b0037c5aaea32fe1da1af553355'})) {
    $matching=@($earlierInventory.gists|Where-Object {$_.facts.saved_ranking_row.entity_id -eq $identity.id})
    if($matching.Count -ne 1 -or $matching[0].facts.source_files.Count -ne 1){throw "Ambiguous earlier source mapping: $($identity.key)"}
    $file=$matching[0].facts.source_files[0]
    $body=Pin $file.verification.local_path $file.manifest_entry.sha256
    $originalSourceMap[$identity.key]=@{path=$body.path;sha256=$body.sha256;source_url=$file.manifest_entry.source_url;inventory=$earlierInventoryPin.path}
}
# Keep successful and FAILED acquisition receipts alike. A retained receipt is not a passed test.
$receipts = @(Get-ChildItem -LiteralPath (Full (Join-Path $AcquisitionDirectory 'receipts')) -Filter '*.json' -File |
    Sort-Object Name | ForEach-Object { Pin $_.FullName })
foreach ($file in Get-ChildItem -LiteralPath (Full (Join-Path $AcquisitionDirectory 'cache')) -Filter 'head-*.json' -File) { [void](Pin $file.FullName) }
foreach ($path in @('examples/gist/karpathy/README.md','examples/gist/karpathy/VERIFICATION.md','examples/gist/karpathy/PARITY.md',
    'examples/gist/karpathy/run.cs','examples/gist/NumSharp.GistExamples.csproj',
    'test/NumSharp.Tests/NumSharp.Tests.csproj','test/NumSharp.Tests.Interop/NumSharp.Tests.Interop.csproj',
    'test/NumSharp.Tests.Interop/GistParity.cs','test/NumSharp.Tests.Interop/Pythonic.cs',
    'test/NumSharp.Tests.Interop/InteropTestBase.cs','test/NumSharp.Tests.Interop/PythonSession.cs',
    'test/NumSharp.Tests.Interop/KarpathyOriginalSource.cs')) { [void](Pin $path) }

# Reconcile acquisition bytes, stable fixture bytes, logical resources and helper hashes.
# Runtime SourceIntegrity tests independently hash the actual compiled embedded bytes.
$fixtureManifestPin=Pin 'test/NumSharp.Tests.Interop/Fixtures/Karpathy/sources.json'
$fixtureManifest=Json $fixtureManifestPin.path
if((Rel $fixtureManifest.source_loader) -ne 'test/NumSharp.Tests.Interop/KarpathyOriginalSource.cs'){throw 'Unexpected original-source loader in fixture manifest.'}
$fixtureMap=@{}
foreach($fixture in $fixtureManifest.sources) {
    $key=$fixture.key
    if(-not $originalSourceMap.ContainsKey($key) -or $fixtureMap.ContainsKey($key)){throw "Unknown/duplicate fixture key: $key"}
    $expected=$originalSourceMap[$key]
    if((Rel $fixture.original_source_path) -ne $expected.path -or $fixture.sha256 -ne $expected.sha256 -or $fixture.source_url -ne $expected.source_url){throw "Fixture/acquisition source mapping differs: $key"}
    $fixturePin=Pin $fixture.fixture_path $expected.sha256
    $fixtureMap[$key]=@{fixture=$fixturePin;original=(Pin $expected.path $expected.sha256);record=$fixture}
}
if($fixtureMap.Count -ne $originalSourceMap.Count){throw 'The stable fixture inventory must cover every approved source exactly once.'}
$sourceHelper=Get-Content -LiteralPath (Full 'test/NumSharp.Tests.Interop/KarpathyOriginalSource.cs') -Raw
$helperPins=@{}
foreach($match in [regex]::Matches($sourceHelper,'\["(?<key>\w+)"\]\s*=\s*"(?<hash>[0-9a-fA-F]{64})"')) {
    $key=$match.Groups['key'].Value
    if($helperPins.ContainsKey($key)){throw "Duplicate original-source helper pin: $key"}
    $helperPins[$key]=$match.Groups['hash'].Value.ToLowerInvariant()
}
if($helperPins.Count -ne $originalSourceMap.Count){throw 'Original-source helper inventory does not match the six approved sources.'}
[xml]$testProject=Get-Content -LiteralPath (Full 'test/NumSharp.Tests.Interop/NumSharp.Tests.Interop.csproj') -Raw
$embedded=[Collections.Generic.List[object]]::new();$embeddedKeys=@{}
foreach($resource in $testProject.SelectNodes("//*[local-name()='EmbeddedResource']")) {
    $logical=[string]$resource.GetAttribute('LogicalName')
    if(-not $logical){$logicalNode=$resource.SelectSingleNode("*[local-name()='LogicalName']");if($null -ne $logicalNode){$logical=[string]$logicalNode.InnerText}}
    if(-not $logical.StartsWith('KarpathyOriginal.')){continue}
    foreach($include in ([string]$resource.GetAttribute('Include')).Split(';')) {
        $path=Full (Join-Path 'test/NumSharp.Tests.Interop' $include)
        $logicalName=$logical.Replace('%(Filename)',[IO.Path]::GetFileNameWithoutExtension($path))
        if($logicalName -notmatch '^KarpathyOriginal\.(?<key>\w+)\.py$'){throw "Unsupported embedded-source logical name: $logicalName"}
        $key=$Matches.key
        if(-not $originalSourceMap.ContainsKey($key) -or $embeddedKeys.ContainsKey($key)){throw "Unknown/duplicate embedded source: $key"}
        $expected=$originalSourceMap[$key]
        if((Rel $path) -ne $fixtureMap[$key].fixture.path -or -not $helperPins.ContainsKey($key) -or $helperPins[$key] -ne $expected.sha256){throw "Incorrect embedded fixture resource path/hash mapping: $key"}
        $embeddedKeys[$key]=$true
        $embedded.Add([ordered]@{key=$key;logical_name=$logicalName;fixture=(Pin $path $expected.sha256);original_acquisition=$fixtureMap[$key].original;source_url=$expected.source_url;inventory=$expected.inventory;fixture_manifest=$fixtureManifestPin.path;helper_sha256=$helperPins[$key]})
    }
}
if($embedded.Count -ne $originalSourceMap.Count){throw 'Not every approved original source is embedded exactly once.'}
foreach ($file in Get-ChildItem -LiteralPath (Full 'examples/gist/karpathy') -File | Where-Object { $_.Extension -in @('.cs','.py','.txt') }) { [void](Pin $file.FullName) }
$runnerEvidence = @(Get-ChildItem -LiteralPath (Full $LogDirectory) -File |
    Where-Object { $_.Extension -in @('.json','.txt') -and $_.Name -match 'runner|demo' } |
    Sort-Object Name | ForEach-Object { Pin $_.FullName })

# A source-declared test inventory detects absent DataRows, not merely internally consistent TRX counters.
$methods = @{}
foreach ($directory in @('test/NumSharp.Tests/Examples','test/NumSharp.Tests.Interop')) {
    foreach ($file in Get-ChildItem -LiteralPath (Full $directory) -Filter 'Karpathy*.cs' -File) {
        [void](Pin $file.FullName); $text = Get-Content -LiteralPath $file.FullName -Raw
        if ($text -notmatch '\[TestClass') { continue }
        $ns = [regex]::Match($text,'(?m)^\s*namespace\s+([\w.]+)\s*[;{]').Groups[1].Value
        $class = [regex]::Match($text,'(?m)^\s*public\s+(?:sealed\s+|partial\s+)?class\s+(\w+)').Groups[1].Value
        if (-not $ns -or $class -notlike 'Karpathy*') { throw "Unrecognized test class: $($file.Name)" }
        foreach ($match in [regex]::Matches($text,'(?ms)(?<attrs>(?:\s*\[[^\]]+\]\s*)+)public\s+(?:async\s+)?(?:void|Task)\s+(?<name>\w+)\s*\(')) {
            $attrs=$match.Groups['attrs'].Value
            if ($attrs -notmatch '\[(?:Data)?TestMethod\b') { continue }
            $rows=[regex]::Matches($attrs,'\[DataRow\s*\(').Count
            if ($attrs -match 'DynamicData' -or ($attrs -match '\[DataTestMethod' -and $rows -eq 0)) { throw 'Unknown dynamic-data schema needs an explicit verifier adapter.' }
            $name="$ns.$class.$($match.Groups['name'].Value)"
            if ($methods.ContainsKey($name)) { throw "Ambiguous current source method: $name" }
            $categories=@([regex]::Matches($attrs,'\bTestCategory\s*\(\s*"([^"\r\n]+)"\s*\)') | ForEach-Object {$_.Groups[1].Value} | Sort-Object -Unique)
            $methods[$name]=@{ file=(Rel $file.FullName); rows=[Math]::Max(1,$rows); data_rows=$rows; categories=$categories; suite=$(if($directory -like '*Interop'){'live'}else{'unit'}) }
        }
    }
}
if ($methods.Count -eq 0) { throw 'No current Karpathy test methods found.' }
$runs=[Collections.Generic.List[object]]::new(); $executions=[Collections.Generic.List[object]]::new(); $runMethods=@{}
foreach ($framework in @('net8.0','net10.0')) { foreach ($suite in @('unit','live')) {
    $log=Pin (Join-Path $LogDirectory "karpathy-$suite-$($framework.Replace('.0',''))-$FinalLogSuffix.trx")
    [xml]$xml=Get-Content -LiteralPath (Full $log.path) -Raw
    $definitions=@($xml.SelectNodes("/*[local-name()='TestRun']/*[local-name()='TestDefinitions']/*[local-name()='UnitTest']"))
    $results=@($xml.SelectNodes("/*[local-name()='TestRun']/*[local-name()='Results']/*[local-name()='UnitTestResult']"))
    $summary=$xml.SelectSingleNode("/*[local-name()='TestRun']/*[local-name()='ResultSummary']")
    $counters=$summary.SelectSingleNode("*[local-name()='Counters']")
    $expected=0
    foreach($declared in $methods.Values) { if($declared.suite -eq $suite) { $expected += $declared.rows } }
    if ($summary.outcome -ne 'Completed' -or $definitions.Count -ne $expected -or $results.Count -ne $expected -or
        [int]$counters.total -ne $expected -or [int]$counters.executed -ne $expected -or [int]$counters.passed -ne $expected) {
        throw "Incomplete run $($log.path): source declares $expected executions."
    }
    $byId=@{}; $seen=@{}; $rowsByMethod=@{}
    foreach ($definition in $definitions) {
        $method=$definition.SelectSingleNode("*[local-name()='TestMethod']")
        $name="$($method.className).$($method.name)"; $id=[string]$definition.id
        if ($byId.ContainsKey($id) -or -not $methods.ContainsKey($name) -or $methods[$name].suite -ne $suite) { throw "Unknown/ambiguous definition: $name" }
        if ($method.codeBase.Replace('\','/') -notmatch "/$([regex]::Escape($framework))/") { throw "Framework mismatch: $name" }
        $loggedCategories=@($definition.SelectNodes("*[local-name()='TestCategory']/*[local-name()='TestCategoryItem']")|ForEach-Object {$_.GetAttribute('TestCategory')}|Sort-Object -Unique)
        if([string]::Join('|',$loggedCategories) -cne [string]::Join('|',$methods[$name].categories)){throw "Current source/TRX category mismatch: $name"}
        $byId[$id]=@{ name=$name; definition=$definition; assembly=[string]$method.codeBase }
    }
    foreach ($result in $results) {
        $id=[string]$result.testId
        if (-not $byId.ContainsKey($id) -or $seen.ContainsKey($id)) { throw "Missing/duplicate result definition: $id" }
        if ($result.outcome -ne 'Passed') { throw "Failed/skipped result: $($result.testName) ($($result.outcome))" }
        if ($byId[$id].definition.SelectSingleNode("*[local-name()='Execution']").id -ne $result.executionId) { throw "Execution id mismatch: $id" }
        $seen[$id]=$true; $name=$byId[$id].name
        $row=[ordered]@{ key="$framework/$suite/$($result.executionId)"; method=$name; name=[string]$result.testName;
            outcome=[string]$result.outcome; framework=$framework; suite=$suite; log=$log.path; test_id=$id;
            execution_id=[string]$result.executionId; duration=[string]$result.duration;categories=$methods[$name].categories;
            started_at=[string]$result.startTime; finished_at=[string]$result.endTime; computer=[string]$result.computerName }
        $executions.Add($row)
        if (-not $rowsByMethod.ContainsKey($name)) { $rowsByMethod[$name]=[Collections.Generic.List[object]]::new() }
        $rowsByMethod[$name].Add($row)
    }
    foreach ($name in $methods.Keys) {
        if ($methods[$name].suite -ne $suite) { continue }
        if (-not $rowsByMethod.ContainsKey($name) -or $rowsByMethod[$name].Count -ne $methods[$name].rows) { throw "Missing method/DataRows: $name on $framework" }
        if (@($rowsByMethod[$name] | ForEach-Object {$_.name} | Sort-Object -Unique).Count -ne $methods[$name].rows) { throw "Duplicate DataRow names: $name" }
    }
    $runMethods["$framework/$suite"]=$rowsByMethod
    $runs.Add([ordered]@{ framework=$framework; suite=$suite; passed=$results.Count; source_declared_executions=$expected;
        log=$log; current_assembly_pins=@($byId.Values.assembly | Sort-Object -Unique | ForEach-Object {Pin $_}) })
} }

# Prior top-ten regressions are separate evidence, never added to the new Karpathy total.
$regressions=[Collections.Generic.List[object]]::new()
foreach($file in Get-ChildItem -LiteralPath (Full $LogDirectory) -Filter 'prior-gist-*-regression.trx' -File | Sort-Object Name) {
    $pin=Pin $file.FullName; [xml]$xml=Get-Content -LiteralPath $file.FullName -Raw
    $results=@($xml.TestRun.Results.UnitTestResult); $counters=$xml.TestRun.ResultSummary.Counters
    if($xml.TestRun.ResultSummary.outcome -ne 'Completed' -or $results.Count -ne [int]$counters.total -or
        $results.Count -ne [int]$counters.passed -or @($results|Where-Object {$_.outcome -ne 'Passed'}).Count -ne 0) { throw "Prior Gist regression is not fully passing: $($file.Name)" }
    $regressions.Add([ordered]@{log=$pin;passed=$results.Count;classification='prior_top_ten_regression_not_new_Karpathy_count'})
}

$coveragePins=[Collections.Generic.List[object]]::new(); $cases=[Collections.Generic.List[object]]::new(); $ids=@{}
foreach ($file in Get-ChildItem -LiteralPath (Full $CoverageDirectory) -Filter 'coverage-*.json' -File | Sort-Object Name) {
    $mapPin=Pin $file.FullName; $map=Json $file.FullName; $coveragePins.Add($mapPin)
    foreach ($section in @('cases','exclusions')) {
        if (-not $map.Contains($section)) { continue }
        for ($i=0;$i -lt $map[$section].Count;$i++) {
            $case=$map[$section][$i]; $id=[string]$case.case_id
            if (-not $id -or $ids.ContainsKey($id)) { throw "Missing/duplicate coverage case id: $id" }; $ids[$id]=$true
            $excluded=$section -eq 'exclusions'; $references=@(Value $case 'tests' @())
            if (-not $excluded -and $references.Count -eq 0) { throw "Coverage case has no measurable tests: $id" }
            $bindings=[Collections.Generic.List[object]]::new()
            foreach ($reference in $references) {
                if ($reference -isnot [string]) { throw "Unknown test-reference schema in $id" }
                $names=@($methods.Keys | Where-Object {$_ -eq $reference -or $_.EndsWith('.'+$reference,[StringComparison]::Ordinal)})
                if ($names.Count -ne 1) { throw "Missing/ambiguous reference '$reference' in $id" }
                $name=$names[0]; $suite=$methods[$name].suite
                foreach ($framework in @('net8.0','net10.0')) {
                    $rows=$runMethods["$framework/$suite"][$name]
                    $selected=@($rows)
                    if($case.Contains('data_row')) {
                        $argument=if($case.data_row -is [string]) {'"'+$case.data_row+'"'}else{[string]$case.data_row}
                        $display=$name.Split('.')[-1]+' ('+$argument+')'
                        $selected=@($rows|Where-Object {$_.name -eq $display})
                        if($selected.Count -ne 1){throw "Missing/ambiguous requested DataRow $display for $id on $framework"}
                    }
                    $bindings.Add([ordered]@{ reference=$reference; method=$name; current_test_source=$methods[$name].file;
                        framework=$framework; suite=$suite; all_declared_data_rows_passed=$true;
                        categories=$methods[$name].categories;execution_keys=@($rows | ForEach-Object {$_.key});case_selected_execution_keys=@($selected|ForEach-Object {$_.key}) })
                }
            }
            $cases.Add([ordered]@{ case_id=$id; classification=$(if($excluded){'exclusion_or_explicit_limitation'}else{'referenced_assertions_passed'});
                coverage_map=$mapPin.path; json_pointer="/$section/$i"; original_record=$case; test_bindings=$bindings.ToArray() })
        }
    }
}
if ($coveragePins.Count -lt 5) { throw 'Expected RNN/Pong/LSTM/microGPT and actual-demo coverage maps.' }
$tierNames=@('KarpathyShortRun','KarpathyByteParity','KarpathySourceIntegrity')
$tiers=[Collections.Generic.List[object]]::new()
foreach($tier in $tierNames) {
    $members=@($executions|Where-Object {$_.categories -contains $tier})
    if($members.Count -eq 0){throw "Required verification tier has no executions: $tier"}
    $perRun=[Collections.Generic.List[object]]::new()
    foreach($run in $runs) {
        $count=@($members|Where-Object {$_.framework -eq $run.framework -and $_.suite -eq $run.suite}).Count
        $perRun.Add([ordered]@{framework=$run.framework;suite=$run.suite;executions=$count})
    }
    $tiers.Add([ordered]@{category=$tier;executions=$members.Count;per_run=$perRun.ToArray();execution_keys=@($members|ForEach-Object {$_.key})})
}
$tierUnion=@($executions|Where-Object {@($_.categories|Where-Object {$_ -in $tierNames}).Count -gt 0})
$supplementaryTiers=[Collections.Generic.List[object]]::new()
foreach($selection in @(@{file='tier-short-net10.trx';category='KarpathyShortRun'},@{file='tier-bytes-net10.trx';category='KarpathyByteParity'})) {
    $path=Full (Join-Path $LogDirectory $selection.file)
    if(-not(Test-Path -LiteralPath $path -PathType Leaf)){continue}
    $pin=Pin $path;[xml]$xml=Get-Content -LiteralPath $path -Raw
    $results=@($xml.TestRun.Results.UnitTestResult);$counters=$xml.TestRun.ResultSummary.Counters
    $expected=@($executions|Where-Object {$_.framework -eq 'net10.0' -and $_.suite -eq 'live' -and $_.categories -contains $selection.category})
    if($xml.TestRun.ResultSummary.outcome -ne 'Completed' -or $results.Count -ne $expected.Count -or
        $results.Count -ne [int]$counters.total -or $results.Count -ne [int]$counters.passed -or @($results|Where-Object {$_.outcome -ne 'Passed'}).Count -ne 0){throw "Incomplete supplementary tier run: $($selection.file)"}
    $definitions=@{}
    foreach($definition in $xml.TestRun.TestDefinitions.UnitTest){$definitions[[string]$definition.id]="$($definition.TestMethod.className).$($definition.TestMethod.name)"}
    foreach($result in $results) {
        $matching=@($expected|Where-Object {$_.method -eq $definitions[[string]$result.testId] -and $_.name -eq [string]$result.testName})
        if($matching.Count -ne 1){throw "Supplementary tier result is not in the final tier: $($result.testName)"}
    }
    $supplementaryTiers.Add([ordered]@{category=$selection.category;framework='net10.0';passed=$results.Count;log=$pin;
        started_at=[string]$xml.TestRun.Times.start;finished_at=[string]$xml.TestRun.Times.finish;classification='separate_tier_execution_not_added_to_full_suite_total'})
}
$negative=[Collections.Generic.List[object]]::new()
if ($SelfTest) {
    $temp=Full (Join-Path $LogDirectory ('verifier-probes-'+[Guid]::NewGuid().ToString('N')))
    foreach ($scenario in @('missing_reference','failed_result')) {
        $directory=Join-Path $temp $scenario; $maps=Join-Path $directory 'maps'; $logs=Join-Path $directory 'logs'
        [void][IO.Directory]::CreateDirectory($maps); [void][IO.Directory]::CreateDirectory($logs)
        foreach($file in Get-ChildItem -LiteralPath (Full $CoverageDirectory) -Filter 'coverage-*.json' -File) {Copy-Item -LiteralPath $file.FullName -Destination $maps}
        foreach($file in Get-ChildItem -LiteralPath (Full $LogDirectory) -Filter "karpathy-*-$FinalLogSuffix.trx" -File) {Copy-Item -LiteralPath $file.FullName -Destination $logs}
        # Acquisition remains separately read-only; only the test maps/logs are negative fixtures.
        if($scenario -eq 'missing_reference') {
            $path=Join-Path $maps 'coverage-rnn.json'; $copy=Json $path; $copy.cases[0].tests[0]='KarpathyMissingClass.NotARealMethod'
            [IO.File]::WriteAllText($path,($copy|ConvertTo-Json -Depth 60),$utf8)
        } else {
            $path=Join-Path $logs "karpathy-unit-net10-$FinalLogSuffix.trx"; [xml]$copy=Get-Content -LiteralPath $path -Raw
            $copy.TestRun.Results.UnitTestResult[0].SetAttribute('outcome','Failed'); $copy.Save($path)
        }
        $reason=$null
        try { & $PSCommandPath -RepositoryRoot $repo -LogDirectory $logs -AcquisitionDirectory $AcquisitionDirectory -FinalLogSuffix $FinalLogSuffix -CoverageDirectory $maps -OutputPath (Join-Path $directory 'unexpected-success.json') }
        catch {$reason=$_.Exception.Message}
        if(-not $reason){throw "Negative fixture unexpectedly passed: $scenario"}
        $negative.Add([ordered]@{scenario=$scenario;rejected=$true;reason=$reason;fixture=(Rel $directory)})
    }
}
$environment=[ordered]@{ observed_at_utc=[DateTime]::UtcNow.ToString('o'); os=[Runtime.InteropServices.RuntimeInformation]::OSDescription;
    architecture=[Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString(); powershell=$PSVersionTable.PSVersion.ToString();
    dotnet_sdk=((& dotnet --version)-join "`n").Trim(); dotnet_runtimes=@(& dotnet --list-runtimes);
    git_head=((& git -C $repo rev-parse HEAD)-join "`n").Trim(); test_computers=@($executions|ForEach-Object {$_.computer}|Sort-Object -Unique); python=$null }
if($PythonExecutable) {
    $details=& $PythonExecutable -c 'import sys,json,numpy,scipy;print(json.dumps({"executable":sys.executable,"python":sys.version,"numpy":numpy.__version__,"scipy":scipy.__version__}))'
    if($LASTEXITCODE -ne 0){throw 'Python environment probe failed.'}; $environment.python=$details|ConvertFrom-Json -AsHashtable
}
$verifier=Pin $PSCommandPath
foreach($path in @($pins.Keys)){[void](Pin $path $pins[$path].sha256)}
$report=[ordered]@{schema_version=1;verified_at_utc=[DateTime]::UtcNow.ToString('o');status='passed';filter='FullyQualifiedName~.Karpathy';
    total_executions=$executions.Count;failed_or_skipped=0;coverage_cases=$cases.Count;
    asserted_cases=@($cases|Where-Object {$_.classification -eq 'referenced_assertions_passed'}).Count;
    exclusions=@($cases|Where-Object {$_.classification -eq 'exclusion_or_explicit_limitation'}).Count;
    scope='Full numerical algorithms and bounded explicitly specified training/inference fixtures; exclusions are not passed claims.';
    parity_claim='Successful execution verifies the contracts actually asserted in the pinned test sources. Exact dtype/shape/byte gates, scalar-value bounds and gradient-check tolerances remain distinct; no exactness is inferred merely from a test name.';
    pin_limitation='Current code/binary hashes are observations at verification time, not proof by themselves of historical build lineage. Raw-file revisions remain distinct from separately verified gist revisions.';
    verifier=$verifier;source_inventory=$inventoryPin;earlier_source_inventory=$earlierInventoryPin;acquisition=$inventory.acquisition;sources=$sources.ToArray();
    acquisition_directory=(Rel $AcquisitionDirectory);log_directory=(Rel $LogDirectory);final_log_suffix=$FinalLogSuffix;
    embedded_original_sources=$embedded.ToArray();original_fixture_manifest=$fixtureManifestPin;
    acquisition_receipts_retained_not_test_results=$receipts;runner_evidence=$runnerEvidence;environment=$environment;
    coverage_maps=$coveragePins.ToArray();runs=$runs.ToArray();executions=$executions.ToArray();cases=$cases.ToArray();
    declared_test_methods=@($methods.Keys|Sort-Object|ForEach-Object {[ordered]@{method=$_;file=$methods[$_].file;suite=$methods[$_].suite;data_rows=$methods[$_].data_rows;categories=$methods[$_].categories}});
    tier_execution_counts=$tiers.ToArray();unique_executions_in_named_tiers=$tierUnion.Count;
    supplementary_tier_runs_not_in_total=$supplementaryTiers.ToArray();
    executions_outside_named_tiers=$executions.Count-$tierUnion.Count;
    tier_counting_policy='Category counts can overlap. Count distinct execution_keys for a union; never sum the tier counts as the total. Declared method categories were checked against each TRX TestDefinition.';
    prior_gist_regressions_not_in_new_total=$regressions.ToArray();existing_port_context=$inventory.existing_ports;
    negative_checks=$negative.ToArray();file_pins=@($pins.Keys|Sort-Object|ForEach-Object {$pins[$_]})}
if(-not(Test-Path -LiteralPath (Split-Path -Parent $outputFull) -PathType Container)){throw 'Output directory must exist.'}
[IO.File]::WriteAllText($outputFull,($report|ConvertTo-Json -Depth 60)+"`n",$utf8)
Write-Host "Verified $($executions.Count) passing executions and $($cases.Count) coverage records. Report: $(Rel $outputFull)"
