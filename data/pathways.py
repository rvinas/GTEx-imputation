def select_genes_pathway(symbols, pathway):
    if pathway == '' or pathway is None:
        gs = symbols  # Returning all symbols
    elif pathway == 'p53':
        gs = ['AIFM2', 'APAF1', 'ATM', 'ATR', 'BAX', 'BBC3', 'BCL2', 'BCL2L1',
              'BID', 'CASP3', 'CASP8', 'CASP9', 'CCNB1', 'CCNB2', 'CCND1',
              'CCND2', 'CCND3', 'CCNE1', 'CCNE2', 'CCNG1', 'CCNG2', 'CD82',
              'CDK1', 'CDK2', 'CDK4', 'CDK6', 'CDKN1A', 'CDKN2A', 'CHEK1',
              'CHEK2', 'CYCS', 'DDB2', 'EI24', 'FAS', 'GADD45A', 'GADD45B',
              'GADD45G', 'GORAB', 'GTSE1', 'IGFBP3', 'MDM2', 'MDM4', 'PERP',
              'PMAIP1', 'PPM1D', 'PTEN', 'RCHY1', 'RRM2B', 'SERPINE1', 'SESN1',
              'SESN2', 'SESN3', 'SFN', 'SHISA5', 'SIAH1', 'SIVA1', 'STEAP3',
              'THBS1', 'TNFRSF10A', 'TNFRSF10B', 'TP53', 'TP53I3', 'TP73',
              'TSC2', 'ZMAT3']
    elif pathway == 'Alzheimer':
        gs = ['ADAM10', 'ADAM17', 'ADRM1', 'AGER', 'AKT1', 'AKT2', 'AKT3',
              'AMBRA1', 'APAF1', 'APBB1', 'APC', 'APH1A', 'APH1B', 'APOE', 'APP',
              'ARAF', 'ATF4', 'ATF6', 'ATG13', 'ATG14', 'ATG2A', 'ATG2B',
              'ATP2A1', 'ATP2A2', 'ATP2A3', 'AXIN1', 'AXIN2', 'BACE1', 'BACE2',
              'BAD', 'BECN1', 'BID', 'BRAF', 'CALM1', 'CALM2', 'CALM3', 'CALML4',
              'CALML5', 'CALML6', 'CAPN1', 'CAPN2', 'CASP3', 'CASP7', 'CASP8',
              'CASP9', 'CDK5', 'CDK5R1', 'CHUK', 'COX4I1', 'COX5A', 'COX5B',
              'COX6A1', 'COX6B1', 'COX6C', 'COX7A1', 'COX7A2', 'COX7A2L',
              'COX7B', 'COX7C', 'COX8A', 'CSF1', 'CSNK1A1', 'CSNK1E', 'CSNK2A1',
              'CSNK2A2', 'CSNK2B', 'CTNNB1', 'CYC1', 'CYCS', 'DDIT3', 'DVL1',
              'DVL2', 'DVL3', 'EIF2AK2', 'EIF2AK3', 'EIF2S1', 'ERN1', 'FADD',
              'FAS', 'FRAT1', 'FRAT2', 'FZD1', 'FZD2', 'FZD3', 'FZD4', 'FZD5',
              'FZD6', 'FZD7', 'FZD8', 'FZD9', 'GAPDH', 'GNAQ', 'GRIN2C',
              'GRIN2D', 'GSK3B', 'HRAS', 'HSD17B10', 'IDE', 'IKBKB', 'IL6',
              'INSR', 'IRS1', 'IRS2', 'ITPR1', 'ITPR2', 'ITPR3', 'KIF5A',
              'KIF5B', 'KIF5C', 'KLC1', 'KLC2', 'KLC4', 'KRAS', 'LPL', 'LRP1',
              'LRP5', 'LRP6', 'MAP2K1', 'MAP2K2', 'MAP2K7', 'MAP3K5', 'MAPK1',
              'MAPK3', 'MAPK8', 'MAPK9', 'MAPT', 'MCU', 'MME', 'MTOR', 'NAE1',
              'NCSTN', 'NDUFA1', 'NDUFA10', 'NDUFA11', 'NDUFA12', 'NDUFA13',
              'NDUFA2', 'NDUFA3', 'NDUFA4', 'NDUFA4L2', 'NDUFA5', 'NDUFA6',
              'NDUFA8', 'NDUFA9', 'NDUFAB1', 'NDUFB1', 'NDUFB10', 'NDUFB11',
              'NDUFB2', 'NDUFB3', 'NDUFB4', 'NDUFB5', 'NDUFB6', 'NDUFB7',
              'NDUFB8', 'NDUFB9', 'NDUFC1', 'NDUFC2', 'NDUFS1', 'NDUFS2',
              'NDUFS3', 'NDUFS4', 'NDUFS5', 'NDUFS6', 'NDUFS7', 'NDUFS8',
              'NDUFV1', 'NDUFV2', 'NDUFV3', 'NFKB1', 'NRAS', 'NRBF2', 'PIK3C3',
              'PIK3CA', 'PIK3CB', 'PIK3CD', 'PIK3R1', 'PIK3R2', 'PIK3R3',
              'PIK3R4', 'PLCB2', 'PLCB3', 'PPID', 'PPIF', 'PPP3CA', 'PPP3CB',
              'PPP3CC', 'PPP3R1', 'PSEN1', 'PSEN2', 'PSENEN', 'PSMA1', 'PSMA2',
              'PSMA3', 'PSMA4', 'PSMA5', 'PSMA6', 'PSMA7', 'PSMB1', 'PSMB2',
              'PSMB4', 'PSMB5', 'PSMB6', 'PSMB7', 'PSMC1', 'PSMC2', 'PSMC3',
              'PSMC4', 'PSMC5', 'PSMC6', 'PSMD1', 'PSMD11', 'PSMD12', 'PSMD13',
              'PSMD14', 'PSMD2', 'PSMD3', 'PSMD4', 'PSMD6', 'PSMD7', 'PSMD8',
              'PSMD9', 'PTGS2', 'RAF1', 'RB1CC1', 'RELA', 'RTN3', 'RTN4', 'SDHA',
              'SDHB', 'SDHC', 'SDHD', 'SLC25A4', 'SLC25A5', 'SLC25A6', 'SNCA',
              'TNF', 'TNFRSF1A', 'TRAF2', 'TUBA1A', 'TUBA1B', 'TUBA1C', 'TUBA3D',
              'TUBA4A', 'TUBA8', 'TUBB', 'TUBB1', 'TUBB2A', 'TUBB2B', 'TUBB4A',
              'TUBB4B', 'TUBB6', 'ULK1', 'ULK2', 'UQCR10', 'UQCR11', 'UQCRB',
              'UQCRC1', 'UQCRC2', 'UQCRFS1', 'UQCRH', 'UQCRQ', 'VDAC1', 'VDAC2',
              'VDAC3', 'WIPI1', 'WIPI2', 'WNT10B', 'WNT11', 'WNT2B', 'WNT3',
              'WNT5B', 'XBP1']
    else:
        raise ValueError('Pathway {} not recognised'.format(pathway))
    gene_idxs = [i for i, g in enumerate(symbols) if g in gs]

    return gene_idxs, gs
