# #390: How to create an NDArray from pointer and NPTypeCode?

- **URL:** https://github.com/SciSharp/NumSharp/issues/390
- **State:** OPEN
- **Author:** @LarryThermo
- **Created:** 2020-01-24T19:52:51Z
- **Updated:** 2020-01-27T17:13:48Z

## Description

First of all thanks for this project.

My code calls into lower level C++ code and I wish to represent the returned array as an NDArray in C#. I have a pointer to the data, the length of the data in bytes, as well as type code. How would I go about creating an NDArray for this case? Something like:

NDArray CreateNDArray(void* data,int dataLengthBytes,NPTypeCode typeCode)
{
var storage = new UnmanagedStorage(   "something here" );
var nda = new NDArray(storage);
return nda;
}

Thanks,

Lars

## Comments

### Comment 1 by @Oceania2018 (2020-01-24T23:19:00Z)

@LarryThermo This [snippet of code](https://github.com/SciSharp/SharpCV/blob/792c8744856cff69f06ddfb17c2866d4d18a05db/src/SharpCV/Core/Mat.cs#L128) will help you.

### Comment 2 by @LarryThermo (2020-01-27T17:13:48Z)

Thank you. It was helpful.

Although this works for me I was hoping to find a solution that did not involve maintaining a case statement for all possible types, and instead took in a void* pointer and an NPTypeCode.

From: Haiping [mailto:notifications@github.com]
Sent: Friday, January 24, 2020 3:19 PM
To: SciSharp/NumSharp <NumSharp@noreply.github.com>
Cc: Rystrom, Larry <Larry.Rystrom@thermofisher.com>; Mention <mention@noreply.github.com>
Subject: Re: [SciSharp/NumSharp] How to create an NDArray from pointer and NPTypeCode? (#390)

CAUTION: This email originated from outside of Thermo Fisher Scientific. If you believe it to be suspicious, report using the Report Phish button in Outlook or send to SOC@thermofisher.com<mailto:SOC@thermofisher.com>.


@LarryThermo<https://urldefense.proofpoint.com/v2/url?u=https-3A__github.com_LarryThermo&d=DwMCaQ&c=q6k2DsTcEGCcCb_WtVSz6hhIl8hvYssy7sH8ZwfbbKU&r=HYF8_GLywj_CyctTtBeHSUbdP3vNJRlUPrSvd7uyd9o&m=ldPFDjE4dQk0khAa1Qi3wOEZBkbjT8L6EAyK7HDruZo&s=a2DXbWNMcxPC_TztnWcAT5iT8MtgAUT-mH2lENF0pdE&e=> This snippet of code<https://urldefense.proofpoint.com/v2/url?u=https-3A__github.com_SciSharp_SharpCV_blob_792c8744856cff69f06ddfb17c2866d4d18a05db_src_SharpCV_Core_Mat.cs-23L128&d=DwMCaQ&c=q6k2DsTcEGCcCb_WtVSz6hhIl8hvYssy7sH8ZwfbbKU&r=HYF8_GLywj_CyctTtBeHSUbdP3vNJRlUPrSvd7uyd9o&m=ldPFDjE4dQk0khAa1Qi3wOEZBkbjT8L6EAyK7HDruZo&s=1Lvm_MMqopf-gcAI9EOGnt1j6tQK0FwjqQ18wrRZVeE&e=> will help you.

—
You are receiving this because you were mentioned.
Reply to this email directly, view it on GitHub<https://urldefense.proofpoint.com/v2/url?u=https-3A__github.com_SciSharp_NumSharp_issues_390-3Femail-5Fsource-3Dnotifications-26email-5Ftoken-3DAESR4HFF7TZJZMWKFYSFKQTQ7NZOLA5CNFSM4KLLHOP2YY3PNVWWK3TUL52HS4DFVREXG43VMVBW63LNMVXHJKTDN5WW2ZLOORPWSZGOEJ4MHJI-23issuecomment-2D578339749&d=DwMCaQ&c=q6k2DsTcEGCcCb_WtVSz6hhIl8hvYssy7sH8ZwfbbKU&r=HYF8_GLywj_CyctTtBeHSUbdP3vNJRlUPrSvd7uyd9o&m=ldPFDjE4dQk0khAa1Qi3wOEZBkbjT8L6EAyK7HDruZo&s=WId0onxeYDYfM4hBRB8PzuM28mnNEBITWmd2RDuB6DQ&e=>, or unsubscribe<https://urldefense.proofpoint.com/v2/url?u=https-3A__github.com_notifications_unsubscribe-2Dauth_AESR4HB3EWNL5XSNOOMUQBLQ7NZOLANCNFSM4KLLHOPQ&d=DwMCaQ&c=q6k2DsTcEGCcCb_WtVSz6hhIl8hvYssy7sH8ZwfbbKU&r=HYF8_GLywj_CyctTtBeHSUbdP3vNJRlUPrSvd7uyd9o&m=ldPFDjE4dQk0khAa1Qi3wOEZBkbjT8L6EAyK7HDruZo&s=Wqb_4KvmYTfef5UzDZbsJ251EHzvAmFEpjzFtTqbgp8&e=>.

