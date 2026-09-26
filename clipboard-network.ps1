# Incoming bridge from clipboard_network.sh clipboards-thru-stdin 
# This deliberately uses a length-prefixed UTF-8 protocol rather than newline-delimited data. So:
# - newlines survive
# - tabs survive
# - Unicode survives
# - emoji survive
# - whitespace survives
# - there is no shell quoting of clipboard contents
# - clipboard data never becomes a command

param(
    [ValidateSet("server", "send", "get")]
    [string]$Mode,

    [int]$Port = 37421
)

$ErrorActionPreference = "Stop"

$StateDir = Join-Path $env:USERPROFILE ".clipboard-network"
$TokenFile = Join-Path $StateDir "token"

New-Item -ItemType Directory -Force $StateDir | Out-Null

if (!(Test-Path $TokenFile)) {
    $token = [Convert]::ToBase64String(
        [Security.Cryptography.RandomNumberGenerator]::GetBytes(32)
    )
    [IO.File]::WriteAllText($TokenFile, $token)
} else {
    $token = [IO.File]::ReadAllText($TokenFile).Trim()
}

function Read-Exact {
    param(
        [System.IO.Stream]$Stream,
        [int]$Count
    )

    $buffer = New-Object byte[] $Count
    $offset = 0

    while ($offset -lt $Count) {
        $n = $Stream.Read($buffer, $offset, $Count - $offset)
        if ($n -le 0) {
            throw "Unexpected end of stream"
        }
        $offset += $n
    }

    return $buffer
}

function Read-Int64 {
    param([System.IO.Stream]$Stream)

    $bytes = Read-Exact $Stream 8
    return [BitConverter]::ToInt64($bytes, 0)
}

function Write-Int64 {
    param(
        [System.IO.Stream]$Stream,
        [Int64]$Value
    )

    $bytes = [BitConverter]::GetBytes($Value)
    $Stream.Write($bytes, 0, $bytes.Length)
}

function Read-String {
    param([System.IO.Stream]$Stream)

    $length = Read-Int64 $Stream

    if ($length -lt 0 -or $length -gt 100000000) {
        throw "Invalid payload length: $length"
    }

    $bytes = Read-Exact $Stream ([int]$length)
    return [Text.Encoding]::UTF8.GetString($bytes)
}

function Write-String {
    param(
        [System.IO.Stream]$Stream,
        [string]$Value
    )

    $bytes = [Text.Encoding]::UTF8.GetBytes($Value)
    Write-Int64 $Stream $bytes.Length
    $Stream.Write($bytes, 0, $bytes.Length)
}

if ($Mode -eq "server") {
    $listener = [Net.Sockets.TcpListener]::new(
        [Net.IPAddress]::Loopback,
        $port
    )

    $listener.Start()

    try {
        while ($true) {
            $client = $listener.AcceptTcpClient()

            try {
                $stream = $client.GetStream()

                $receivedToken = Read-String $stream

                if ($receivedToken -cne $token) {
                    throw "Authentication failed"
                }

                $operation = Read-String $stream

                switch ($operation) {
                    "set" {
                        $text = Read-String $stream

                        Set-Clipboard -Value $text

                        Write-String $stream "ok"
                    }

                    "get" {
                        $text = Get-Clipboard -Raw

                        if ($null -eq $text) {
                            $text = ""
                        }

                        Write-String $stream $text
                    }

                    default {
                        throw "Unknown operation: $operation"
                    }
                }
            }
            catch {
                try {
                    Write-String $stream ("error: " + $_.Exception.Message)
                } catch {}
            }
            finally {
                $stream.Close()
                $client.Close()
            }
        }
    }
    finally {
        $listener.Stop()
    }

    exit
}

# ---------------------------------------------------------------------------
# SSH-side client
# ---------------------------------------------------------------------------

$tcp = [Net.Sockets.TcpClient]::new()
$tcp.Connect("0.0.0.0", $Port)
# $tcp.Connect("127.0.0.1", $Port)

try {
    $stream = $tcp.GetStream()

    Write-String $stream $token
    Write-String $stream $Mode

    if ($Mode -eq "send") {
        # SSH stdin -> desktop-session bridge
        $input = [Console]::OpenStandardInput()
        $memory = New-Object IO.MemoryStream

        $input.CopyTo($memory)

        $text = [Text.Encoding]::UTF8.GetString($memory.ToArray())

        Write-String $stream $text

        $result = Read-String $stream

        if ($result -ne "ok") {
            throw $result
        }
    }
    elseif ($Mode -eq "get") {
        $text = Read-String $stream

        [Console]::OpenStandardOutput().Write(
            [Text.Encoding]::UTF8.GetBytes($text),
            0,
            [Text.Encoding]::UTF8.GetByteCount($text)
        )
    }
}
finally {
    $stream.Close()
    $tcp.Close()
}