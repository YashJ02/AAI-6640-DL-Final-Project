param(
    [string]$OutputPath = "e:\Personal Projects\AAI-6640-DL-Final-Project\AAI6640_Intraday_Project_Animated.pptx"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Add-Type -AssemblyName System.Drawing
Add-Type -AssemblyName Microsoft.Office.Interop.PowerPoint
Add-Type -AssemblyName office

function New-OleColor {
    param(
        [int]$R,
        [int]$G,
        [int]$B
    )
    return [System.Drawing.ColorTranslator]::ToOle([System.Drawing.Color]::FromArgb($R, $G, $B))
}

function Add-GradientBackground {
    param(
        $Slide,
        $Presentation,
        [int[]]$ColorA,
        [int[]]$ColorB
    )

    $bg = $Slide.Shapes.AddShape(
        [Microsoft.Office.Core.MsoAutoShapeType]::msoShapeRectangle,
        0,
        0,
        $Presentation.PageSetup.SlideWidth,
        $Presentation.PageSetup.SlideHeight
    )
    $bg.Fill.Visible = [Microsoft.Office.Core.MsoTriState]::msoTrue
    $bg.Fill.ForeColor.RGB = New-OleColor -R $ColorA[0] -G $ColorA[1] -B $ColorA[2]
    $bg.Fill.BackColor.RGB = New-OleColor -R $ColorB[0] -G $ColorB[1] -B $ColorB[2]
    $bg.Fill.TwoColorGradient(1, 1)
    $bg.Line.Visible = [Microsoft.Office.Core.MsoTriState]::msoFalse
    $bg.ZOrder([Microsoft.Office.Core.MsoZOrderCmd]::msoSendToBack)
    return $bg
}

function Add-Card {
    param(
        $Slide,
        [single]$Left,
        [single]$Top,
        [single]$Width,
        [single]$Height,
        [int[]]$FillRgb,
        [double]$Transparency = 0.12
    )

    $shape = $Slide.Shapes.AddShape(
        [Microsoft.Office.Core.MsoAutoShapeType]::msoShapeRoundedRectangle,
        $Left,
        $Top,
        $Width,
        $Height
    )
    $shape.Fill.Visible = [Microsoft.Office.Core.MsoTriState]::msoTrue
    $shape.Fill.ForeColor.RGB = New-OleColor -R $FillRgb[0] -G $FillRgb[1] -B $FillRgb[2]
    $shape.Fill.Transparency = $Transparency
    $shape.Line.Visible = [Microsoft.Office.Core.MsoTriState]::msoFalse
    return $shape
}

function Add-TextBox {
    param(
        $Slide,
        [single]$Left,
        [single]$Top,
        [single]$Width,
        [single]$Height,
        [string]$Text,
        [string]$FontName = "Aptos",
        [single]$FontSize = 24,
        [bool]$Bold = $false,
        [int[]]$ColorRgb = @(245, 247, 250),
        [int]$Alignment = 1
    )

    $shape = $Slide.Shapes.AddTextbox(
        [Microsoft.Office.Core.MsoTextOrientation]::msoTextOrientationHorizontal,
        $Left,
        $Top,
        $Width,
        $Height
    )

    $shape.TextFrame2.WordWrap = [Microsoft.Office.Core.MsoTriState]::msoTrue
    $shape.TextFrame2.TextRange.Text = $Text
    $shape.TextFrame2.TextRange.Font.Name = $FontName
    $shape.TextFrame2.TextRange.Font.Size = $FontSize
    $shape.TextFrame2.TextRange.Font.Bold = if ($Bold) { [Microsoft.Office.Core.MsoTriState]::msoTrue } else { [Microsoft.Office.Core.MsoTriState]::msoFalse }
    $shape.TextFrame2.TextRange.Font.Fill.ForeColor.RGB = New-OleColor -R $ColorRgb[0] -G $ColorRgb[1] -B $ColorRgb[2]
    $shape.TextFrame2.TextRange.ParagraphFormat.Alignment = $Alignment

    return $shape
}

function Set-SlideTransition {
    param($Slide)

    $Slide.SlideShowTransition.EntryEffect = [Microsoft.Office.Interop.PowerPoint.PpEntryEffect]::ppEffectFadeSmoothly
    $Slide.SlideShowTransition.Speed = [Microsoft.Office.Interop.PowerPoint.PpTransitionSpeed]::ppTransitionSpeedMedium
    $Slide.SlideShowTransition.AdvanceOnClick = [Microsoft.Office.Core.MsoTriState]::msoTrue
    $Slide.SlideShowTransition.AdvanceOnTime = [Microsoft.Office.Core.MsoTriState]::msoFalse
}

function Add-EntryExitAnimations {
    param(
        [System.Collections.ArrayList]$Shapes,
        [Microsoft.Office.Interop.PowerPoint.PpEntryEffect]$Effect = [Microsoft.Office.Interop.PowerPoint.PpEntryEffect]::ppEffectAppear
    )

    $order = 1
    foreach ($shape in $Shapes) {
        $anim = $shape.AnimationSettings
        $anim.Animate = [Microsoft.Office.Core.MsoTriState]::msoTrue
        $anim.EntryEffect = $Effect
        $anim.AnimationOrder = $order
        $anim.AdvanceMode = [Microsoft.Office.Interop.PowerPoint.PpAdvanceMode]::ppAdvanceOnTime
        $anim.AdvanceTime = [Math]::Min(2.4, 0.18 * $order)
        # Hide on click gives presenters a clean slide exit sequence before transition.
        $anim.AfterEffect = [Microsoft.Office.Interop.PowerPoint.PpAfterEffect]::ppAfterEffectHideOnClick
        $order++
    }
}

$ppt = $null
$presentation = $null

try {
    $ppt = New-Object -ComObject PowerPoint.Application
    $ppt.Visible = [Microsoft.Office.Core.MsoTriState]::msoTrue

    $presentation = $ppt.Presentations.Add([Microsoft.Office.Core.MsoTriState]::msoTrue)
    $presentation.PageSetup.SlideSize = [Microsoft.Office.Interop.PowerPoint.PpSlideSizeType]::ppSlideSizeOnScreen16x9

    $bgA = @(15, 23, 42)
    $bgB = @(17, 74, 95)
    $card = @(255, 255, 255)
    $accent = @(255, 196, 84)
    $mint = @(102, 232, 196)
    $ink = @(12, 20, 36)

    # Slide 1: Title
    $slide = $presentation.Slides.Add(1, [Microsoft.Office.Interop.PowerPoint.PpSlideLayout]::ppLayoutBlank)
    [void](Add-GradientBackground -Slide $slide -Presentation $presentation -ColorA $bgA -ColorB $bgB)
    $s1Shapes = [System.Collections.ArrayList]::new()
    [void]$s1Shapes.Add((Add-TextBox -Slide $slide -Left 70 -Top 90 -Width 1120 -Height 120 -Text "Intraday Direction Intelligence Platform" -FontName "Aptos Display" -FontSize 56 -Bold $true))
    [void]$s1Shapes.Add((Add-TextBox -Slide $slide -Left 74 -Top 200 -Width 980 -Height 58 -Text "AAI 6640 Final Project | Multi-Architecture Deep Learning System" -FontName "Aptos" -FontSize 24 -ColorRgb $mint))
    [void]$s1Shapes.Add((Add-TextBox -Slide $slide -Left 75 -Top 312 -Width 760 -Height 120 -Text "Team Members`nOm Patel`nYash Jain`nRuthvik Bandari" -FontName "Aptos" -FontSize 28 -Bold $true))
    [void](Add-Card -Slide $slide -Left 820 -Top 318 -Width 370 -Height 170 -FillRgb $card -Transparency 0.08)
    [void]$s1Shapes.Add((Add-TextBox -Slide $slide -Left 846 -Top 350 -Width 320 -Height 105 -Text "30 S&P500 equities`n40 engineered features`n4 model contenders" -FontName "Aptos" -FontSize 24 -Bold $true -ColorRgb $accent -Alignment 2))
    Set-SlideTransition -Slide $slide
    Add-EntryExitAnimations -Shapes $s1Shapes

    # Slide 2: Scope and scale
    $slide = $presentation.Slides.Add(2, [Microsoft.Office.Interop.PowerPoint.PpSlideLayout]::ppLayoutBlank)
    [void](Add-GradientBackground -Slide $slide -Presentation $presentation -ColorA @(10, 34, 59) -ColorB @(20, 83, 104))
    $s2Shapes = [System.Collections.ArrayList]::new()
    [void]$s2Shapes.Add((Add-TextBox -Slide $slide -Left 60 -Top 40 -Width 1150 -Height 70 -Text "Project Scope and Systems Complexity" -FontName "Aptos Display" -FontSize 44 -Bold $true))
    [void](Add-Card -Slide $slide -Left 70 -Top 130 -Width 540 -Height 500 -FillRgb $card -Transparency 0.1)
    [void]$s2Shapes.Add((Add-TextBox -Slide $slide -Left 95 -Top 160 -Width 500 -Height 420 -Text "Data + Universe`n- 30 equities across 5 sectors`n- 58-day intraday history at 5-minute bars`n- 130,534 supervised rows after cleaning and labeling`n- Related context symbols: SPY, QQQ, VIX, IWM, DIA, TLT, GLD, DXY`n`nEngineering Outputs`n- Full audit reports per ticker`n- Reproducible raw cache + artifact pipeline" -FontName "Aptos" -FontSize 21 -ColorRgb $ink))
    [void](Add-Card -Slide $slide -Left 650 -Top 130 -Width 540 -Height 500 -FillRgb $card -Transparency 0.1)
    [void]$s2Shapes.Add((Add-TextBox -Slide $slide -Left 675 -Top 160 -Width 500 -Height 420 -Text "Modeling + Validation`n- 3 deep architectures + soft-voting ensemble`n- Session-aware labels with volatility normalization`n- Walk-forward sessions and optional time-aware K-fold CV`n- Focal loss, warmup-cosine scheduling, class-bias decision tuning`n`nEvaluation + Delivery`n- McNemar significance tests`n- Volatility-regime and backtesting analytics`n- Streamlit dashboard over generated artifacts" -FontName "Aptos" -FontSize 21 -ColorRgb $ink))
    Set-SlideTransition -Slide $slide
    Add-EntryExitAnimations -Shapes $s2Shapes

    # Slide 3: Pipeline architecture
    $slide = $presentation.Slides.Add(3, [Microsoft.Office.Interop.PowerPoint.PpSlideLayout]::ppLayoutBlank)
    [void](Add-GradientBackground -Slide $slide -Presentation $presentation -ColorA @(12, 20, 32) -ColorB @(20, 55, 90))
    $s3Shapes = [System.Collections.ArrayList]::new()
    [void]$s3Shapes.Add((Add-TextBox -Slide $slide -Left 60 -Top 40 -Width 1150 -Height 70 -Text "End-to-End Architecture" -FontName "Aptos Display" -FontSize 44 -Bold $true))

    $pipelineCards = @(
        @{L=70;  T=145; W=220; H=110; Txt="1) Data Ingestion`n(yfinance + cache)"},
        @{L=330; T=145; W=220; H=110; Txt="2) Cleaning`n(session + anomaly guards)"},
        @{L=590; T=145; W=220; H=110; Txt="3) Features`n(40-dim representation)"},
        @{L=850; T=145; W=300; H=110; Txt="4) Labels`nEWMA-normalized 3-class targets"},
        @{L=70;  T=310; W=220; H=110; Txt="5) Splits`nwalk-forward / time-aware kfold"},
        @{L=330; T=310; W=220; H=110; Txt="6) Training`nAMP + focal + warmup cosine"},
        @{L=590; T=310; W=220; H=110; Txt="7) Evaluation`nmetrics + McNemar + regimes"},
        @{L=850; T=310; W=300; H=110; Txt="8) Productization`nartifacts + dashboard + reports"}
    )

    foreach ($item in $pipelineCards) {
        [void](Add-Card -Slide $slide -Left $item.L -Top $item.T -Width $item.W -Height $item.H -FillRgb $card -Transparency 0.1)
        [void]$s3Shapes.Add((Add-TextBox -Slide $slide -Left ($item.L + 12) -Top ($item.T + 14) -Width ($item.W - 24) -Height ($item.H - 20) -Text $item.Txt -FontName "Aptos" -FontSize 18 -Bold $true -ColorRgb $ink -Alignment 2))
    }

    [void]$s3Shapes.Add((Add-TextBox -Slide $slide -Left 72 -Top 470 -Width 1080 -Height 120 -Text "Complexity highlight: the system is not a single model script; it is a controlled research stack with leakage prevention, data governance, model benchmarking, and portfolio-level analytics." -FontName "Aptos" -FontSize 23 -ColorRgb @(228, 236, 246)))
    Set-SlideTransition -Slide $slide
    Add-EntryExitAnimations -Shapes $s3Shapes

    # Slide 4: Data quality discipline
    $slide = $presentation.Slides.Add(4, [Microsoft.Office.Interop.PowerPoint.PpSlideLayout]::ppLayoutBlank)
    [void](Add-GradientBackground -Slide $slide -Presentation $presentation -ColorA @(20, 25, 45) -ColorB @(44, 62, 80))
    $s4Shapes = [System.Collections.ArrayList]::new()
    [void]$s4Shapes.Add((Add-TextBox -Slide $slide -Left 60 -Top 40 -Width 1150 -Height 70 -Text "Data Quality and Leakage Controls" -FontName "Aptos Display" -FontSize 44 -Bold $true))

    [void](Add-Card -Slide $slide -Left 70 -Top 130 -Width 560 -Height 500 -FillRgb $card -Transparency 0.1)
    [void](Add-Card -Slide $slide -Left 660 -Top 130 -Width 530 -Height 500 -FillRgb $card -Transparency 0.1)

    [void]$s4Shapes.Add((Add-TextBox -Slide $slide -Left 95 -Top 165 -Width 520 -Height 420 -Text "Cleaning Rules`n- Duplicate timestamp elimination`n- NY regular-session filtering`n- OHLC consistency + volume validity checks`n- Session-level outlier clipping (returns + intrabar range)`n- Low-coverage session drop policy`n`nObserved Impact`n- Average row-drop rate: 1.315%`n- Maximum ticker drop: MSFT at 4.343%" -FontName "Aptos" -FontSize 22 -ColorRgb $ink))

    [void]$s4Shapes.Add((Add-TextBox -Slide $slide -Left 685 -Top 165 -Width 485 -Height 420 -Text "Leakage Prevention`n- Returns computed with session boundaries`n- Train-only normalization statistics`n- Optional time-aware fold generation`n- Fold-adaptive thresholding from train split only`n`nGovernance Artifacts`n- ticker_cleaning_report.csv`n- ticker_modeling_report.csv`n- modeling_summary.json" -FontName "Aptos" -FontSize 22 -ColorRgb $ink))

    Set-SlideTransition -Slide $slide
    Add-EntryExitAnimations -Shapes $s4Shapes

    # Slide 5: Feature and label stack
    $slide = $presentation.Slides.Add(5, [Microsoft.Office.Interop.PowerPoint.PpSlideLayout]::ppLayoutBlank)
    [void](Add-GradientBackground -Slide $slide -Presentation $presentation -ColorA @(22, 45, 68) -ColorB @(32, 94, 117))
    $s5Shapes = [System.Collections.ArrayList]::new()
    [void]$s5Shapes.Add((Add-TextBox -Slide $slide -Left 60 -Top 40 -Width 1150 -Height 70 -Text "Feature Stack + Label Intelligence" -FontName "Aptos Display" -FontSize 44 -Bold $true))

    [void](Add-Card -Slide $slide -Left 70 -Top 130 -Width 360 -Height 470 -FillRgb $card -Transparency 0.1)
    [void](Add-Card -Slide $slide -Left 450 -Top 130 -Width 360 -Height 470 -FillRgb $card -Transparency 0.1)
    [void](Add-Card -Slide $slide -Left 830 -Top 130 -Width 360 -Height 470 -FillRgb $card -Transparency 0.1)

    [void]$s5Shapes.Add((Add-TextBox -Slide $slide -Left 90 -Top 160 -Width 320 -Height 410 -Text "Stationary Market Signals`n- log_return`n- hl_range`n- oc_body`n- upper_shadow`n- volume_log_change`n`nGoal: remove price-level drift and preserve intrabar behavior." -FontName "Aptos" -FontSize 20 -ColorRgb $ink))
    [void]$s5Shapes.Add((Add-TextBox -Slide $slide -Left 470 -Top 160 -Width 320 -Height 410 -Text "Technical + Time Context`n- 18 indicators (RSI, MACD, ATR, ADX, etc.)`n- Fourier intraday cycles (sin/cos)`n- Day-of-week encoding`n- Related-market features from 6 external symbols in this run`n`nTotal active feature columns: 40" -FontName "Aptos" -FontSize 20 -ColorRgb $ink))
    [void]$s5Shapes.Add((Add-TextBox -Slide $slide -Left 850 -Top 160 -Width 320 -Height 410 -Text "Volatility-Normalized Labels`n- Future log returns`n- EWMA sigma (lambda=0.94)`n- z-return thresholds mapped to down/neutral/up`n`nCurrent run distribution`n- Down: 3.13%`n- Neutral: 93.99%`n- Up: 2.88%" -FontName "Aptos" -FontSize 20 -ColorRgb $ink))

    Set-SlideTransition -Slide $slide
    Add-EntryExitAnimations -Shapes $s5Shapes

    # Slide 6: Model portfolio
    $slide = $presentation.Slides.Add(6, [Microsoft.Office.Interop.PowerPoint.PpSlideLayout]::ppLayoutBlank)
    [void](Add-GradientBackground -Slide $slide -Presentation $presentation -ColorA @(20, 22, 55) -ColorB @(43, 56, 92))
    $s6Shapes = [System.Collections.ArrayList]::new()
    [void]$s6Shapes.Add((Add-TextBox -Slide $slide -Left 60 -Top 40 -Width 1150 -Height 70 -Text "Model Architecture Portfolio" -FontName "Aptos Display" -FontSize 44 -Bold $true))

    [void](Add-Card -Slide $slide -Left 70 -Top 130 -Width 360 -Height 490 -FillRgb $card -Transparency 0.1)
    [void](Add-Card -Slide $slide -Left 450 -Top 130 -Width 360 -Height 490 -FillRgb $card -Transparency 0.1)
    [void](Add-Card -Slide $slide -Left 830 -Top 130 -Width 360 -Height 490 -FillRgb $card -Transparency 0.1)

    [void]$s6Shapes.Add((Add-TextBox -Slide $slide -Left 95 -Top 155 -Width 310 -Height 440 -Text "LSTM + Temporal Attention`n- 3 stacked recurrent layers`n- LayerNorm + attention weighting over timesteps`n- Context vector classifier head`n- Interpretable attention maps" -FontName "Aptos" -FontSize 21 -Bold $true -ColorRgb $ink))
    [void]$s6Shapes.Add((Add-TextBox -Slide $slide -Left 475 -Top 155 -Width 310 -Height 440 -Text "Temporal Fusion Transformer`n- Feature gating via VSN`n- Ticker embedding as static covariate`n- GRN blocks + multi-head attention`n- Native route to feature importance extraction" -FontName "Aptos" -FontSize 21 -Bold $true -ColorRgb $ink))
    [void]$s6Shapes.Add((Add-TextBox -Slide $slide -Left 855 -Top 155 -Width 310 -Height 440 -Text "Dilated CNN + BiLSTM`n- Parallel dilations (1,2,4) for receptive field growth`n- BatchNorm + ReLU conv stack`n- Bidirectional LSTM sequence aggregation`n- Captures local and medium-range microstructure patterns" -FontName "Aptos" -FontSize 21 -Bold $true -ColorRgb $ink))

    [void]$s6Shapes.Add((Add-TextBox -Slide $slide -Left 110 -Top 588 -Width 980 -Height 40 -Text "Meta-layer: weighted soft-voting ensemble with fold-level validation-weighted probabilities" -FontName "Aptos" -FontSize 18 -ColorRgb @(230, 237, 248) -Alignment 2))
    Set-SlideTransition -Slide $slide
    Add-EntryExitAnimations -Shapes $s6Shapes

    # Slide 7: Training engine
    $slide = $presentation.Slides.Add(7, [Microsoft.Office.Interop.PowerPoint.PpSlideLayout]::ppLayoutBlank)
    [void](Add-GradientBackground -Slide $slide -Presentation $presentation -ColorA @(16, 35, 35) -ColorB @(31, 85, 85))
    $s7Shapes = [System.Collections.ArrayList]::new()
    [void]$s7Shapes.Add((Add-TextBox -Slide $slide -Left 60 -Top 40 -Width 1150 -Height 70 -Text "Training Engine and Operations" -FontName "Aptos Display" -FontSize 44 -Bold $true))

    [void](Add-Card -Slide $slide -Left 70 -Top 130 -Width 560 -Height 500 -FillRgb $card -Transparency 0.1)
    [void](Add-Card -Slide $slide -Left 660 -Top 130 -Width 530 -Height 500 -FillRgb $card -Transparency 0.1)

    [void]$s7Shapes.Add((Add-TextBox -Slide $slide -Left 95 -Top 165 -Width 520 -Height 420 -Text "Optimization Stack`n- Focal loss (gamma=2.0) with optional class weighting`n- Label smoothing and gradient clipping`n- AdamW + warmup-cosine schedule`n- Automatic mixed precision on CUDA`n- Early stopping on validation macro-F1`n`nDecision Control`n- Bias-grid class decision tuning`n- Objective-selectable (macro-F1 or accuracy)" -FontName "Aptos" -FontSize 21 -ColorRgb $ink))

    [void]$s7Shapes.Add((Add-TextBox -Slide $slide -Left 685 -Top 165 -Width 485 -Height 420 -Text "Experiment Operations`n- Checkpointing per fold and model`n- Epoch-level training logs for overfit diagnostics`n- MLflow experiment logging`n- Fold-wise prediction persistence (.npz)`n- KPI gate enforcement layer`n`nThis is a production-like MLOps loop, not just notebook training." -FontName "Aptos" -FontSize 21 -ColorRgb $ink))

    Set-SlideTransition -Slide $slide
    Add-EntryExitAnimations -Shapes $s7Shapes

    # Slide 8: Evaluation rigor
    $slide = $presentation.Slides.Add(8, [Microsoft.Office.Interop.PowerPoint.PpSlideLayout]::ppLayoutBlank)
    [void](Add-GradientBackground -Slide $slide -Presentation $presentation -ColorA @(28, 22, 37) -ColorB @(73, 46, 89))
    $s8Shapes = [System.Collections.ArrayList]::new()
    [void]$s8Shapes.Add((Add-TextBox -Slide $slide -Left 60 -Top 40 -Width 1150 -Height 70 -Text "Evaluation Rigor" -FontName "Aptos Display" -FontSize 44 -Bold $true))

    [void](Add-Card -Slide $slide -Left 70 -Top 130 -Width 360 -Height 500 -FillRgb $card -Transparency 0.1)
    [void](Add-Card -Slide $slide -Left 450 -Top 130 -Width 360 -Height 500 -FillRgb $card -Transparency 0.1)
    [void](Add-Card -Slide $slide -Left 830 -Top 130 -Width 360 -Height 500 -FillRgb $card -Transparency 0.1)

    [void]$s8Shapes.Add((Add-TextBox -Slide $slide -Left 90 -Top 165 -Width 320 -Height 430 -Text "Classification Metrics`n- Accuracy`n- Macro F1`n- Per-class precision/recall/F1`n- Confusion matrices`n- Majority baseline comparison" -FontName "Aptos" -FontSize 21 -ColorRgb $ink))
    [void]$s8Shapes.Add((Add-TextBox -Slide $slide -Left 470 -Top 165 -Width 320 -Height 430 -Text "Statistical Testing`n- Pairwise McNemar tests across models`n- Discordant-error analysis`n- p-values tracked per fold`n`nObserved example`n- LSTM vs CNN-LSTM p = 0.0833" -FontName "Aptos" -FontSize 21 -ColorRgb $ink))
    [void]$s8Shapes.Add((Add-TextBox -Slide $slide -Left 850 -Top 165 -Width 320 -Height 430 -Text "Market-Aware Analytics`n- Volatility regime degradation`n- Strategy backtests and risk ratios`n- Confidence-threshold sweeps`n- Sharpe/Sortino/Calmar + max drawdown" -FontName "Aptos" -FontSize 21 -ColorRgb $ink))

    Set-SlideTransition -Slide $slide
    Add-EntryExitAnimations -Shapes $s8Shapes

    # Slide 9: Result snapshot
    $slide = $presentation.Slides.Add(9, [Microsoft.Office.Interop.PowerPoint.PpSlideLayout]::ppLayoutBlank)
    [void](Add-GradientBackground -Slide $slide -Presentation $presentation -ColorA @(12, 30, 50) -ColorB @(18, 58, 84))
    $s9Shapes = [System.Collections.ArrayList]::new()
    [void]$s9Shapes.Add((Add-TextBox -Slide $slide -Left 60 -Top 40 -Width 1150 -Height 70 -Text "Current Run Snapshot (accuracy_90 profile)" -FontName "Aptos Display" -FontSize 42 -Bold $true))

    [void](Add-Card -Slide $slide -Left 70 -Top 130 -Width 1120 -Height 260 -FillRgb $card -Transparency 0.08)
    [void]$s9Shapes.Add((Add-TextBox -Slide $slide -Left 95 -Top 160 -Width 1080 -Height 210 -Text "Model             Accuracy   Macro-F1   Delta vs Baseline`nLSTM              0.9367     0.3224     0.0000`nCNN-LSTM          0.9365     0.3224    -0.0002`nTFT               0.9367     0.3224     0.0000`nSoft Ensemble     0.9367     0.3224     0.0000" -FontName "Consolas" -FontSize 24 -ColorRgb $ink))

    [void](Add-Card -Slide $slide -Left 70 -Top 420 -Width 540 -Height 205 -FillRgb $card -Transparency 0.1)
    [void](Add-Card -Slide $slide -Left 650 -Top 420 -Width 540 -Height 205 -FillRgb $card -Transparency 0.1)

    [void]$s9Shapes.Add((Add-TextBox -Slide $slide -Left 95 -Top 445 -Width 500 -Height 165 -Text "Interpretation`n- Very high neutral-class prevalence drives accuracy`n- Macro-F1 reveals class-boundary difficulty`n- Confusion patterns motivate richer decision objectives" -FontName "Aptos" -FontSize 20 -ColorRgb $ink))
    [void]$s9Shapes.Add((Add-TextBox -Slide $slide -Left 675 -Top 445 -Width 500 -Height 165 -Text "Why this still matters`n- The benchmarking framework exposed this behavior quickly`n- The pipeline supports alternate configs (kfold/adaptive)`n- Architecture + infra are ready for iterative label redesign" -FontName "Aptos" -FontSize 20 -ColorRgb $ink))

    Set-SlideTransition -Slide $slide
    Add-EntryExitAnimations -Shapes $s9Shapes

    # Slide 10: Trading outcomes
    $slide = $presentation.Slides.Add(10, [Microsoft.Office.Interop.PowerPoint.PpSlideLayout]::ppLayoutBlank)
    [void](Add-GradientBackground -Slide $slide -Presentation $presentation -ColorA @(25, 46, 20) -ColorB @(67, 110, 45))
    $s10Shapes = [System.Collections.ArrayList]::new()
    [void]$s10Shapes.Add((Add-TextBox -Slide $slide -Left 60 -Top 40 -Width 1150 -Height 70 -Text "Backtest and Portfolio Impact" -FontName "Aptos Display" -FontSize 44 -Bold $true))

    [void](Add-Card -Slide $slide -Left 70 -Top 130 -Width 560 -Height 500 -FillRgb $card -Transparency 0.1)
    [void](Add-Card -Slide $slide -Left 660 -Top 130 -Width 530 -Height 500 -FillRgb $card -Transparency 0.1)

    [void]$s10Shapes.Add((Add-TextBox -Slide $slide -Left 95 -Top 165 -Width 520 -Height 430 -Text "CNN-LSTM Strategy (best observed)`n- Sharpe: 4.6888`n- Sortino: 11.2525`n- Calmar: 60.4045`n- Final equity: 1.0114`n- Max drawdown: -0.0068`n`nBenchmark for same horizon`n- Final equity: 1.0028`n- Sharpe: 0.5795" -FontName "Aptos" -FontSize 23 -ColorRgb $ink))

    [void]$s10Shapes.Add((Add-TextBox -Slide $slide -Left 685 -Top 165 -Width 485 -Height 430 -Text "Threshold Sweep Discipline`n- Confidence thresholds tested from 0.50 to 0.95`n- Metrics tracked per threshold: sharpe, sortino, calmar, final equity, drawdown`n- Robust artifacts exported for every model family`n`nTakeaway`n- Architecture selection can materially change tradability, even when headline classification metrics look similar." -FontName "Aptos" -FontSize 22 -ColorRgb $ink))

    Set-SlideTransition -Slide $slide
    Add-EntryExitAnimations -Shapes $s10Shapes

    # Slide 11: Productization and UX
    $slide = $presentation.Slides.Add(11, [Microsoft.Office.Interop.PowerPoint.PpSlideLayout]::ppLayoutBlank)
    [void](Add-GradientBackground -Slide $slide -Presentation $presentation -ColorA @(30, 28, 24) -ColorB @(80, 66, 40))
    $s11Shapes = [System.Collections.ArrayList]::new()
    [void]$s11Shapes.Add((Add-TextBox -Slide $slide -Left 60 -Top 40 -Width 1150 -Height 70 -Text "Product Layer and Reproducibility" -FontName "Aptos Display" -FontSize 44 -Bold $true))

    [void](Add-Card -Slide $slide -Left 70 -Top 130 -Width 360 -Height 500 -FillRgb $card -Transparency 0.1)
    [void](Add-Card -Slide $slide -Left 450 -Top 130 -Width 360 -Height 500 -FillRgb $card -Transparency 0.1)
    [void](Add-Card -Slide $slide -Left 830 -Top 130 -Width 360 -Height 500 -FillRgb $card -Transparency 0.1)

    [void]$s11Shapes.Add((Add-TextBox -Slide $slide -Left 90 -Top 165 -Width 320 -Height 430 -Text "CLI Workflow`n- mode data`n- mode train`n- mode full`n- optional model subsets`n- config-driven execution" -FontName "Aptos" -FontSize 21 -ColorRgb $ink))
    [void]$s11Shapes.Add((Add-TextBox -Slide $slide -Left 470 -Top 165 -Width 320 -Height 430 -Text "Artifact Contract`n- checkpoints`n- fold predictions`n- training histories`n- KPI reports`n- data quality reports`n- evaluation summaries" -FontName "Aptos" -FontSize 21 -ColorRgb $ink))
    [void]$s11Shapes.Add((Add-TextBox -Slide $slide -Left 850 -Top 165 -Width 320 -Height 430 -Text "Presentation UX`n- Streamlit dashboard`n- model comparison charts`n- confusion matrices`n- prediction overlays`n- confidence diagnostics`n- backtest views" -FontName "Aptos" -FontSize 21 -ColorRgb $ink))

    Set-SlideTransition -Slide $slide
    Add-EntryExitAnimations -Shapes $s11Shapes

    # Slide 12: Roadmap and closing
    $slide = $presentation.Slides.Add(12, [Microsoft.Office.Interop.PowerPoint.PpSlideLayout]::ppLayoutBlank)
    [void](Add-GradientBackground -Slide $slide -Presentation $presentation -ColorA @(9, 21, 33) -ColorB @(6, 54, 76))
    $s12Shapes = [System.Collections.ArrayList]::new()
    [void]$s12Shapes.Add((Add-TextBox -Slide $slide -Left 60 -Top 40 -Width 1150 -Height 70 -Text "Roadmap: From Strong Platform to Stronger Alpha" -FontName "Aptos Display" -FontSize 44 -Bold $true))

    [void](Add-Card -Slide $slide -Left 70 -Top 130 -Width 1120 -Height 390 -FillRgb $card -Transparency 0.1)
    [void]$s12Shapes.Add((Add-TextBox -Slide $slide -Left 95 -Top 170 -Width 1070 -Height 320 -Text "Next Technical Steps`n- Rebalance label policy to reduce neutral dominance while preserving market realism`n- Expand temporal context with higher-frequency and cross-session memory modules`n- Add probabilistic calibration and class-conditional threshold optimization`n- Integrate transaction-cost-aware training objectives`n- Extend significance testing across additional folds and market regimes`n`nThis project already delivers a complete research-to-product loop, and it is structured for rapid iteration." -FontName "Aptos" -FontSize 24 -ColorRgb $ink))

    [void]$s12Shapes.Add((Add-TextBox -Slide $slide -Left 90 -Top 552 -Width 1100 -Height 70 -Text "Om Patel | Yash Jain | Ruthvik Bandari" -FontName "Aptos" -FontSize 26 -Bold $true -ColorRgb $accent -Alignment 2))

    Set-SlideTransition -Slide $slide
    Add-EntryExitAnimations -Shapes $s12Shapes

    if (Test-Path -Path $OutputPath) {
        Remove-Item -Path $OutputPath -Force
    }

    $presentation.SaveAs($OutputPath)
    Write-Output "Presentation created: $OutputPath"
}
finally {
    if ($presentation -ne $null) {
        $presentation.Close()
        [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($presentation)
    }
    if ($ppt -ne $null) {
        $ppt.Quit()
        [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($ppt)
    }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}
