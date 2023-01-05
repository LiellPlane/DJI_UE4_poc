// Fill out your copyright notice in the Description page of Project Settings.

#include "IVILO_COMMS.h"
//**************************************
//INCLUDE files for :A simple TakeScreenShot() Function
//**************************************

#include <Objbase.h>
#include <wincodec.h>
#define WIN32_LEAN_AND_MEAN
#include "Windows/MinWindows.h"
#include "Containers/UnrealString.h"
#include "Math/UnrealMathUtility.h"
#include <Winerror.h>
#include <iostream>
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "HAL/PlatformFilemanager.h"
#include "GenericPlatform/GenericPlatformFile.h"
#include <fstream>
#include <stdio.h>

// Sets default values for this component's properties
UIVILO_COMMS::UIVILO_COMMS()
{
	// Set this component to be initialized when the game starts, and to be ticked every frame.  You can turn these features
	// off to improve performance if you don't need them.
	PrimaryComponentTick.bCanEverTick = true;

	// ...
}

// Called when the game starts
void UIVILO_COMMS::BeginPlay()
{
	Super::BeginPlay();
	// ...

}

// Called every frame
void UIVILO_COMMS::TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction)
{
	Super::TickComponent(DeltaTime, TickType, ThisTickFunction);
	//UE_LOG(LogTemp, Error, TEXT("im alive!!"));
	// ...

}

//handle user request for screenshot
void UIVILO_COMMS::TakeDeskTopImage(FString &FilePath_out, bool &isOK_out)
{

	if (FrameDivider < 0 || FrameDivider>100)
	{
		FrameDivider = 5;

	}


	if (UpdateCounter >= FrameDivider)
	{
		UpdateCounter = 0;
		bool charse;
		//charse = TakeImageSaveBitmap(L"C:/Users/3DVisLab/Desktop/liell2.bmp");
		charse = TakeImageSaveBitmap(L"Z:/liell2.bmp", Grab_XShift);
		FilePath_out = "file:///Z:/liell2.bmp";//in form for UE download image from URL node
		if (charse == FALSE)
		{
			UE_LOG(LogTemp, Error, TEXT("failure with TakeImageSaveBitmap function"));
		}
		isOK_out = charse;
	}

	else

	{
		UpdateCounter = UpdateCounter + 1;
		FilePath_out = "EMPTY";
		isOK_out = FALSE;
	}
}

void UIVILO_COMMS::GetAssetPositions(float & latitude, float & longitude, float & heading, float &height, float & gimbalpitch, float & speed, int32 & LastUpdate_sec, FString &AssetID, bool & isError)
{
	//update blueprint on request with latest position info
	latitude = AssetInfoCard.latitude;
	longitude = AssetInfoCard.longitude;
	heading = AssetInfoCard.heading;
	gimbalpitch = AssetInfoCard.GimbalPitch;
	height = AssetInfoCard.altitude;
	LastUpdate_sec = GetWorld()->GetRealTimeSeconds() - AssetInfoCard.LastUpdated;
	AssetID = AssetInfoCard.NameOfAsset;
	speed = AssetInfoCard.speed;
}

void UIVILO_COMMS::ListenForReqCommand(bool & IsReady, bool& NewInputCommand, EUnitCommands & RequestedCommand, float & latitude, float & longitude, float & heading, float & height, float & gimbalpitch, float & Speed, bool & isTurnClockWise, int32 & LastUpdate_sec, FString & AssetID, bool & isError)
{///listen for IVILO format input command matching my ID then update output
	//fields 
	//fix user input invalid updatecounter 
	//check if listener is busy - if its busy then
	//function has recieved valid input and is waiting for which response to send
	LastUpdate_sec = GetWorld()->GetRealTimeSeconds() - InputPosReq.LastUpdated;
	IsReady = CommandManager.Listener_Ready;


	if (CommandHandler_FrameDivider < 0 || CommandHandler_FrameDivider >100)
	{
		CommandHandler_FrameDivider = 30;
	}
	//check if active frame (randint to distribute update load for multiple instances of object)
	if (CommandManager.RequestManager_UpdateCounter >= CommandHandler_FrameDivider + RandInt)
	{
		RandInt = FMath::RandRange(0, CommonSet.RandomRange);//update random addition to update rate
		CommandManager.RequestManager_UpdateCounter = 0;


		SaveLivePositionStatus(AssetReqCard, MailBox_InfoCard);


		if (CommandManager.Listener_Ready == false)
		{
			ASSET_SpotLightColour.InitFromString(CommonSet.ColourORANGE);
			return;
		}


		ASSET_SpotLightColour.InitFromString(CommonSet.ColourGREEN);
		//first listen for command 
		//now listen for response
		FString clipboardtext = MailBox_Check(MyMailBoxType, MailBox_InfoCard, MyID);

		//FString clipboardtext = GetString_OSClipboard();
		//quick first check - if same message ignore
		if (CommandManager.Listener_LastClipBoardMsg == clipboardtext)
		{
			NewInputCommand = false;
			isError = false;
			//UE_LOG(LogTemp, Error, TEXT(" %s ignored duplicate message:  %s"), *MyID, *clipboardtext);
			return;
		}
		//initialise info cards for recording input request
		FAssetInfoCard NewRequestCard;
		FParseInputMessage ParseResult = ParseInputMsg(clipboardtext, MyID, NewRequestCard);
		CommandManager.Listener_LastClipBoardMsg = clipboardtext;

		//now check results of ParseResult
		if (ParseResult.isForMe == true && ParseResult.isError == false)
		{
			//check is a new ticket number
			if (InputPosReq.TicketNumber == ParseResult.ParsedTicket)
			{
				NewInputCommand = false;
				isError = false;
				UE_LOG(LogTemp, Error, TEXT("ignored input msg, old ticket:  %s"), *MyID);
				return;
			}

			NewInputCommand = true;
			isError = false;
			//update global variable position request card
			InputPosReq.InputCommand = ParseResult.ParsedCommand;
			InputPosReq.latitude = NewRequestCard.latitude;
			//UE_LOG(LogTemp, Error, TEXT("PASS message latitude: %f"), InputPosReq.latitude);
			InputPosReq.longitude = NewRequestCard.longitude;
			InputPosReq.heading = NewRequestCard.heading;
			InputPosReq.altitude = NewRequestCard.altitude;
			InputPosReq.GimbalPitch = NewRequestCard.GimbalPitch;
			InputPosReq.speed = NewRequestCard.speed;
			InputPosReq.isTurningClockwise = NewRequestCard.isTurningClockwise;
			InputPosReq.LastUpdated = NewRequestCard.LastUpdated;
			InputPosReq.NameOfAsset = ParseResult.TargetID;
			InputPosReq.TicketNumber = ParseResult.ParsedTicket;
			InputPosReq.UnParsedInputString = clipboardtext;
			InputPosReq.OnReq_AckFinished = false;
			InputPosReq.OnReq_ErrMessage = CommonSet.EMPTY;
			InputPosReq.UnParsedInputString = "";
			CommandManager.Listener_Ready = false;
			ASSET_SpotLightColour.InitFromString(CommonSet.ColourORANGE);
			//update outputs
			RequestedCommand = InputPosReq.InputCommand;
			latitude = InputPosReq.latitude;
			longitude = InputPosReq.longitude;
			heading = InputPosReq.heading;
			height = InputPosReq.altitude;
			gimbalpitch = InputPosReq.GimbalPitch;
			Speed = InputPosReq.speed;
			isTurnClockWise = InputPosReq.isTurningClockwise;
			AssetID = InputPosReq.NameOfAsset;

			//UE_LOG(LogTemp, Error, TEXT("PASS message found for me %s"), *MyID);
		}
		else if (ParseResult.isForMe == true && ParseResult.isError == true)
		{
			NewInputCommand = true;
			isError = true;
			ASSET_SpotLightColour.InitFromString(CommonSet.ColourRED);
			UE_LOG(LogTemp, Error, TEXT("FAILED message found for me %s"), *MyID);
		}
		else
		{
			NewInputCommand = false;
			isError = false;
		}


	}
	else { CommandManager.RequestManager_UpdateCounter = CommandManager.RequestManager_UpdateCounter + 1; }



}

void UIVILO_COMMS::ReplyClear_ReqCommand(bool isPASS, FString ErrorMsg)
{
	//generates response from current input request from IVILO
	if (InputPosReq.OnReq_AckFinished == true)//check card hasnt been handled already
	{
		return;
	}
	InputPosReq.OnReq_IsPass = isPASS;//set if request has passed or failed
	InputPosReq.OnReq_ErrMessage = ErrorMsg;//set error message if isPass=FALSE
	FString ACK_string = GenerateACKString(InputPosReq);
	MailBox_Send(MyMailBoxType, MailBox_InfoCard, CommonSet.IVILO, ACK_string);

	//SendString_OSClipboard(ACK_string);
	InputPosReq.OnReq_AckFinished = true;
	CommandManager.Listener_Ready = true;//start listening for REQs again
}

//COPY function for windows OS
void UIVILO_COMMS::SendString_OSClipboard(FString InputString)
{
	const FString MyString = InputString;
	// Copy text to clipboard
	FPlatformMisc::ClipboardCopy(*MyString);
}

//PASTE function for windows OS
FString UIVILO_COMMS::GetString_OSClipboard()
{
	FString MyString = "";
	FPlatformMisc::ClipboardPaste(MyString);
	return MyString;
	//UE_LOG(LogTemp, Error, TEXT("pasted clipboard text is %s"),* MyString);
}

//translates input command innumerator into communication string
FString UIVILO_COMMS::GenerateCommandString(EUnitCommands InputCommand, int32 TicketNumber, FString TargetID)
{
	FString output = "";
	//start building common command string according to protocol
	output = TargetID + CommonSet.Delimiter;
	output = output + MyID + CommonSet.Delimiter;
	output = output + FString::FromInt(TicketNumber) + CommonSet.Delimiter;
	output = output + CommonSet.REQ + CommonSet.Delimiter;
	//get string version of enum
	FString StringCommand = ConvertEnumToString(InputCommand);
	output = output + StringCommand + CommonSet.Delimiter;
	output = output + CommonSet.PASS + CommonSet.Delimiter;//make it same format as incoming messages

	//now get specific suffix such as variables etc
	//specific suffix values for CHANGE GIMBAL PITCH
	if (InputCommand == EUnitCommands::ChangeGimbalPitch)
	{
		output = output + GetFloatAsStringWithPrecision(AssetReqCard.GimbalPitch, 2) + CommonSet.Delimiter;
	}
	//specific suffix values for CHANGE POSITION
	if (InputCommand == EUnitCommands::ChangeDroneLatLongHeight)
	{
		//get longitude latitude etc - must match with DJI SDK 
		output = output + GetFloatAsStringWithPrecision(AssetReqCard.GimbalPitch, 2) + CommonSet.Delimiter;
		output = output + GetFloatAsStringWithPrecision(AssetReqCard.altitude, 2) + CommonSet.Delimiter;
		output = output + GetFloatAsStringWithPrecision(AssetReqCard.latitude, 2) + CommonSet.Delimiter;
		output = output + GetFloatAsStringWithPrecision(AssetReqCard.longitude, 2) + CommonSet.Delimiter;
		output = output + GetFloatAsStringWithPrecision(AssetReqCard.heading, 2) + CommonSet.Delimiter;
		output = output + GetFloatAsStringWithPrecision(AssetReqCard.speed, 2) + CommonSet.Delimiter;
		output = output + (AssetReqCard.isClockWiseTurn ? TEXT("1") : TEXT("0")) + CommonSet.Delimiter;
		output = output + "nostatus" + CommonSet.Delimiter;
	}


	//UE_LOG(LogTemp, Error, TEXT("%s generated: %s"),*MyID, *output);
	return output;
}

FString UIVILO_COMMS::GenerateACKString(FAssetInfoCard InputReqCard)
{
	//encode to IVILO protocol the ACK result of a REQ from ivilo
	FString output = "";
	FString TargetID = CommonSet.IVILO;
	//start building common command string according to protocol
	output = TargetID + CommonSet.Delimiter;
	output = output + InputReqCard.NameOfAsset + CommonSet.Delimiter;
	output = output + FString::FromInt(InputReqCard.TicketNumber) + CommonSet.Delimiter;
	output = output + CommonSet.ACK + CommonSet.Delimiter;
	//get string version of enum
	FString StringCommand = ConvertEnumToString(InputReqCard.InputCommand);
	output = output + StringCommand + CommonSet.Delimiter;
	//conditional ACK response
	if (InputReqCard.OnReq_IsPass == true)
	{
		output = output + CommonSet.PASS + CommonSet.Delimiter;//make it same format as incoming messages
	}
	else
	{
		output = output + CommonSet.FAIL + CommonSet.Delimiter;
		//if a fail, add fail message
		output = output + InputReqCard.OnReq_ErrMessage + CommonSet.Delimiter;
	}


	UE_LOG(LogTemp, Error, TEXT("%s ACK generated %s"), *InputReqCard.NameOfAsset, *output);
	return output;
}

FParseInputMessage UIVILO_COMMS::ParseInputMsg(FString InputCommandString, FString MyIDinternal, FAssetInfoCard&OutputInfoCard)
{
	//Parses input protocol, populates info card
	//Can parse OUTPUT (req) messages and INPUT (ack) messages
	//EXAMPLE OF INPUT STRING:
	//DJI1*IVILO*39*REQ*ChangeDroneLatLongHeight*PASS*0*30*56.476*-2.953*180*6.6*0*nostatus*
	//EXAMPLE OF OUTPUT STRING
	//IVILO*DJI1**39*ACK*ChangeDroneLatLongHeight*FAIL*cannot connect to gimbal*

//	UE_LOG(LogTemp, Error, TEXT("%s parsing : %s"),*MyIDinternal,*InputCommandString)

	FParseInputMessage OutputStruct;//structure will be initialised in error
	//delimit string into an array
	TArray<FString> CommandArray;
	OutputStruct.DelimitsFound = InputCommandString.ParseIntoArray(CommandArray, TEXT("*"), true);


	//check enough delimiter characters found
	if (OutputStruct.DelimitsFound < 4)
	{
		//UE_LOG(LogTemp, Error, TEXT("Delimits found too low: %d"), OutputStruct.DelimitsFound);
		return OutputStruct;//will returned error'd structure
	}


	//check delimited array has enough elements
	int32 DelimitedArrayCount = CommandArray.Num();
	if (DelimitedArrayCount < 5)
	{
		//UE_LOG(LogTemp, Error, TEXT("command array too small: %d"), Count);
		return OutputStruct;//will returned error'd structure
	}



	//test target ID
	OutputStruct.TargetID = CommandArray[0];
	if (OutputStruct.TargetID != MyIDinternal)
	{
		//UE_LOG(LogTemp, Error, TEXT("targetID same as me, ignored  %s"), *OutputStruct.TargetID);
		return OutputStruct;//will returned error'd structure
	}
	//message is for me - set output
	OutputStruct.isForMe = true;


	//get caller ID
	OutputStruct.CallerID = CommandArray[1];


	//check ticket number is VALID
	FString TempTicker = CommandArray[2];
	if (TempTicker.IsNumeric() == false)
	{
		UE_LOG(LogTemp, Error, TEXT("%s ; parsed ticket number invalid! : %s"), *MyIDinternal, *CommandArray[2]);
		OutputStruct.isError = true;
		return OutputStruct;//will returned error'd structure
	}
	//convert ticket number WARNING UNSAFE FUNCTION
	FString StringParsedTicket = CommandArray[2];
	OutputStruct.ParsedTicket = FCString::Atoi(*CommandArray[2]);
	if (CommandArray[3] == CommonSet.ACK)
	{
		OutputStruct.isAck_elseReq = true;
	}
	else if (CommandArray[3] == CommonSet.REQ)
	{
		OutputStruct.isAck_elseReq = false;
	}
	else//ERROR! Not an ACK or REQ!
	{
		UE_LOG(LogTemp, Error, TEXT("%s ; parsed command no ACK or REQ found in element! : %s"), *MyIDinternal, *CommandArray[3]);
		OutputStruct.isError = true;
		return OutputStruct;//will returned error'd structure
	}



	//get input command
	EUnitCommands InputCommand = ConvertStringToEnum(CommandArray[4]);
	if (InputCommand == EUnitCommands::FAIL)
	{
		UE_LOG(LogTemp, Error, TEXT("%s; Error parsing command, no match for : %s"), *MyIDinternal, *CommandArray[4]);
	}
	OutputStruct.ParsedCommand = InputCommand;
	FString ConvertedCommandString = ConvertEnumToString(InputCommand);//generate string for debug purposes
//	UE_LOG(LogTemp, Error, TEXT("%s : result is: %s"),*MyIDinternal, *InputCommandString)

		//check success/fail of command
	if (CommandArray[5] == CommonSet.FAIL)
	{
		OutputStruct.isCommandPass = false;
		if (OutputStruct.DelimitsFound > 5)//make sure a fail message exists after FAIL message
		{
			OutputStruct.FailureReason = CommandArray[6];
			//	UE_LOG(LogTemp, Error, TEXT("%s; parsed command result %s for ticket %s is %s"), *MyIDinternal, *ConvertedCommandString, *StringParsedTicket, (OutputStruct.isCommandPass ? TEXT("True") : TEXT("False")));

				//UE_LOG(LogTemp, Error, TEXT("%s; parsed command has reported fail : %s"), *MyIDinternal,*CommandArray[6]);
		}
		else
		{
			OutputStruct.FailureReason = CommonSet.EMPTY;
			//	UE_LOG(LogTemp, Error, TEXT("%s; parsed command result has failed but no reason found"),*MyIDinternal);
		}
		return OutputStruct;//will return structure
	}
	else if (CommandArray[5] == CommonSet.PASS)
	{
		OutputStruct.isCommandPass = true;
	}
	else//ERROR! Not an ACK or REQ!
	{
		UE_LOG(LogTemp, Error, TEXT("%s; parsed command no PASS or FAIL found in element! : %s"), *MyIDinternal, *CommandArray[5]);
		OutputStruct.isError = true;
		return OutputStruct;//will returned error'd structure
	}
	OutputInfoCard.LastUpdated = GetWorld()->GetRealTimeSeconds();
	OutputInfoCard.NameOfAsset = OutputStruct.CallerID;


	//UE_LOG(LogTemp, Error, TEXT("%s; parsing command: %s for ticket %s"), *MyIDinternal, *ConvertedCommandString,*OutputStruct.ParsedTicket)
	//UE_LOG(LogTemp, Error, TEXT("%s; parsed command result %s for ticket %s is %s"), *MyIDinternal, *ConvertedCommandString, *StringParsedTicket, (OutputStruct.isCommandPass ? TEXT("True") : TEXT("False")));
	//now onto conditional suffixes for commands/responses
	FString TempElement = "";
	int32 CurrentIndex = 5;

	//special case for getting/setting position - update asset card rather
	//than synchronously returning state
	//same format for these two input commands/returns
	if (InputCommand == EUnitCommands::ReportAllPosition && OutputStruct.isAck_elseReq == false)//if a REQ there are no elements/values after req, so can return parse now
	{
		return OutputStruct;//
	}
	if (InputCommand == EUnitCommands::ChangeDroneLatLongHeight && OutputStruct.isAck_elseReq == true)//if an ACK there are no elements/values after req, so can return parse now
	{
		return OutputStruct;//
	}
	if (InputCommand == EUnitCommands::ChangeGimbalPitch && OutputStruct.isAck_elseReq == true)//if a ACK there are no elements/values after req, so can return parse now
	{
		return OutputStruct;//
	}
	if (InputCommand == EUnitCommands::StopMission)//ACk or REQ no values needed
	{
		return OutputStruct;//
	}

	if (InputCommand == EUnitCommands::ChangeGimbalPitch && OutputStruct.isAck_elseReq == false)//if a REQ there should be a variable included
	{
		if (DelimitedArrayCount < CurrentIndex + 1)//current variables being sent down
		{
			UE_LOG(LogTemp, Error, TEXT("%s; parsed command, error expected more data for ChangeGimbalPitch"), *MyIDinternal);
			OutputStruct.isError = true;
			return OutputStruct;//will returned error'd structure
		}
		//check is numeric
		TempElement = CommandArray[CurrentIndex + 1];
		if (TempTicker.IsNumeric() == false)
		{
			UE_LOG(LogTemp, Error, TEXT("%s; parsed command, expected number for ChangeGimbalPitch position : %s"), *MyIDinternal, *CommandArray[CurrentIndex + 1]);
			OutputStruct.isError = true;
			return OutputStruct;//will returned error'd structure
		}
		//all checks passed - populate card
		OutputInfoCard.GimbalPitch = FCString::Atof(*CommandArray[CurrentIndex + 1]);
		UE_LOG(LogTemp, Error, TEXT("%s; parsed command, req accepted to move gimbal: %f"), *MyIDinternal, OutputInfoCard.GimbalPitch);

	}


	//same format needed to package these two commands
	if (InputCommand == EUnitCommands::ReportAllPosition || InputCommand == EUnitCommands::ChangeDroneLatLongHeight)
	{


		if (DelimitedArrayCount < CommonSet.ReportAllPosition_DelimitedCount)//current variables being sent down
		{
			UE_LOG(LogTemp, Error, TEXT("%s; parsed command, error expected more data for report all position"), *MyIDinternal);
			OutputStruct.isError = true;
			return OutputStruct;//will returned error'd structure
		}


		bool CheckAllInfoNumeric = true;
		//test all incoming positions are valid
		for (int i = 1; i < 6; i++)
		{
			TempElement = CommandArray[CurrentIndex + i];
			if (TempTicker.IsNumeric() == false)
			{
				UE_LOG(LogTemp, Error, TEXT("%s; parsed command, expected number for position : %s"), *MyIDinternal, *CommandArray[CurrentIndex + i]);
				OutputStruct.isError = true;
				return OutputStruct;//will returned error'd structure
			}
		}

		//order is dictated by DJI SDK 

		OutputInfoCard.GimbalPitch = FCString::Atof(*CommandArray[CurrentIndex + 1]);
		OutputInfoCard.altitude = FCString::Atof(*CommandArray[CurrentIndex + 2]);
		OutputInfoCard.latitude = FCString::Atof(*CommandArray[CurrentIndex + 3]);
		OutputInfoCard.longitude = FCString::Atof(*CommandArray[CurrentIndex + 4]);
		OutputInfoCard.heading = FCString::Atof(*CommandArray[CurrentIndex + 5]);
		OutputInfoCard.speed = FCString::Atof(*CommandArray[CurrentIndex + 6]);

		if (FCString::Atof(*CommandArray[CurrentIndex + 6]) == 0.0f)
		{
			OutputInfoCard.isTurningClockwise = false;
		}
		else
		{
			OutputInfoCard.isTurningClockwise = true;
		}

		OutputInfoCard.Status = CommandArray[CurrentIndex + 8];

	}

	//	UE_LOG(LogTemp, Error, TEXT("%s; parsed finished succesfully : %s"), *MyIDinternal, *ConvertedCommandString)

	return OutputStruct;
}

//convert enum to string - improve this in the future 
FString UIVILO_COMMS::ConvertEnumToString(EUnitCommands InputCommand)
{
	if (InputCommand == EUnitCommands::ChangeDroneLatLongHeight)
	{
		return CommonSet.ChangeDroneLatLongHeight;
	}


	if (InputCommand == EUnitCommands::ReportAllPosition)
	{
		return CommonSet.ReportAllPosition;
	}

	if (InputCommand == EUnitCommands::DemoMission)
	{
		return CommonSet.DemoMission;
	}

	if (InputCommand == EUnitCommands::DoNothing)
	{
		return CommonSet.DoNothing;
	}

	if (InputCommand == EUnitCommands::StopMission)
	{
		return CommonSet.StopMission;
	}

	if (InputCommand == EUnitCommands::ChangeGimbalPitch)
	{
		return CommonSet.ChangeGimbalPitch;
	}
	UE_LOG(LogTemp, Error, TEXT("Error trying to extract command string from input command enum!"));
	return CommonSet.Error;
}

EUnitCommands UIVILO_COMMS::ConvertStringToEnum(FString InputCommand)
{
	if (InputCommand == CommonSet.ReportAllPosition)
	{
		return EUnitCommands::ReportAllPosition;
	}

	if (InputCommand == CommonSet.ChangeGimbalPitch)
	{
		return EUnitCommands::ChangeGimbalPitch;
	}

	if (InputCommand == CommonSet.DoNothing)
	{
		return EUnitCommands::DoNothing;
	}

	if (InputCommand == CommonSet.StopMission)
	{
		return EUnitCommands::StopMission;
	}

	if (InputCommand == CommonSet.ReportAllPosition)
	{
		return EUnitCommands::ReportAllPosition;
	}

	if (InputCommand == CommonSet.ChangeDroneLatLongHeight)
	{
		return EUnitCommands::ChangeDroneLatLongHeight;
	}

	return EUnitCommands::FAIL;
}

bool UIVILO_COMMS::GenerateCommFolders(FString Input_MyID)
{
	isCommFoldersReady = PrepareMyFolders(MailBox_InfoCard, Input_MyID);
	return isCommFoldersReady;
}

void UIVILO_COMMS::CommandHandler(EUnitCommands InputCommand, bool Onhigh_ResetError, FString &out_Response, bool &out_WaitingForAck, bool &out_Ready, bool &out_Error)
{

	//fix user input invalid updatecounter 
	if (CommandHandler_FrameDivider < 0 || CommandHandler_FrameDivider >100)
	{
		CommandHandler_FrameDivider = 30;
	}

	//check if active frame
	if (CommandManager.CommandManager_UpdateCounter >= CommandHandler_FrameDivider + RandInt)
	{
		RandInt = FMath::RandRange(0, CommonSet.RandomRange);//update random addition to update rate
		CommandManager.CommandManager_UpdateCounter = 0;
		//ReadLivePositionStatus("SIM1");//test
		//ReadLivePositionStatus("SIM2", MailBox_InfoCard);//test
		//ReadLivePositionStatus("SIM3", MailBox_InfoCard);//test


		//faulted state
		if (CommandManager.CurrentState == EStateMachine::Faulted)
		{//active high reset input
			UE_LOG(LogTemp, Error, TEXT("%s: faulted"), *MyID);
			IVILO_SpotLightColour.InitFromString(CommonSet.ColourRED);
			CommandManager.Error = true;
			CommandManager.Ready = false;
			if (CommandManager.LastResetErrState == false && Onhigh_ResetError == true)
			{
				UE_LOG(LogTemp, Error, TEXT("%s: reset active"), *MyID);
				(CommandManager.CurrentState = EStateMachine::Reseting);
			}
		}

		//reset state - clear all internal variables
		if (CommandManager.CurrentState == EStateMachine::Reseting)
		{
			UE_LOG(LogTemp, Error, TEXT("%s: Reseting"), *MyID);
			CommandManager.CurrentREQ = "";
			CommandManager.Error = false;
			CommandManager.Ready = true;
			CommandManager.WaitingForACK = false;
			(CommandManager.CurrentState = EStateMachine::Idle);
			UE_LOG(LogTemp, Error, TEXT("%s: Ready"), *MyID);
		}

		//idling state - ready to receive command to send
		if (CommandManager.CurrentState == EStateMachine::Idle)
		{
			IVILO_SpotLightColour.InitFromString(CommonSet.ColourGREEN);
			//UE_LOG(LogTemp, Error, TEXT("cmd: idle"));
			if (InputCommand != EUnitCommands::DoNothing)
			{
				UE_LOG(LogTemp, Error, TEXT("%s: input command found"), *MyID);
				DroneCommandSelect = EUnitCommands::DoNothing;//sets external command to nothing to stop looping instruction
				CommandManager.WaitingForACK = true;
				CommandManager.Ready = false;
				CommandManager.CurrentTicketNumber++;//need to have new ticket each command or will be rejected
				if (CommandManager.CurrentTicketNumber > 999) { CommandManager.CurrentTicketNumber = 1; }
				//generate command string into comms protocol and post into clipboard
				CommandManager.CurrentREQ = GenerateCommandString(InputCommand, CommandManager.CurrentTicketNumber, AssetReqCard.NameOfAsset);
				//populate send receipt
				SentMessageReceipt.SendTicket = CommandManager.CurrentTicketNumber;
				SentMessageReceipt.SentCommand = InputCommand;
				SentMessageReceipt.StartedTime = GetWorld()->GetRealTimeSeconds();
				SentMessageReceipt.TargetID = AssetReqCard.NameOfAsset;
				//send string to clipboard (current communication method)
				bool mailOK = MailBox_Send(AssetReqCard.MailBoxType, MailBox_InfoCard, AssetReqCard.NameOfAsset, CommandManager.CurrentREQ);
				if (mailOK == false)
				{
					(CommandManager.CurrentState = EStateMachine::Faulted);
				}
				else
				{
					//SendString_OSClipboard(CommandManager.CurrentREQ);
				//UE_LOG(LogTemp, Error, TEXT("cmd: input command sent %s"),*CommandManager.CurrentREQ);
					(CommandManager.CurrentState = EStateMachine::Waiting);//now waiting for response

				}
			}
		}

		//after sending a command - wait for response
		if (CommandManager.CurrentState == EStateMachine::Waiting)
		{

			IVILO_SpotLightColour.InitFromString(CommonSet.ColourORANGE);
			//UE_LOG(LogTemp, Error, TEXT("cmd: Waiting"));
			//check for time out (no ACK response from REQ command)
			double elapsedtime = GetWorld()->GetRealTimeSeconds() - SentMessageReceipt.StartedTime;
			if (elapsedtime > CommandACK_TimeOut_Secs)
			{
				UE_LOG(LogTemp, Error, TEXT("%s: time out"), *MyID);
				CommandManager.WaitingForACK = true;
				CommandManager.Ready = false;
				CommandManager.Error = true;
				CommandManager.CurrentState = EStateMachine::Faulted;
			}

			//now listen for response
			FString clipboardtext = MailBox_Check(AssetReqCard.MailBoxType, MailBox_InfoCard, MyID);
			//FString clipboardtext = GetString_OSClipboard();
			if (clipboardtext != CommandManager.LastClipBoardMsg)
			{
				FParseInputMessage ParseResult = ParseInputMsg(clipboardtext, MyID, AssetInfoCard);
				CommandManager.LastClipBoardMsg = clipboardtext;
				//now check results of PARSE RESULT
				if (ParseResult.isForMe == true && ParseResult.isError == false)
				{
					CommandManager.Ready = true;
					CommandManager.Error = false;
					CommandManager.CurrentState = EStateMachine::Idle;
				}
				else if (ParseResult.isForMe == true && ParseResult.isError == true)
				{
					CommandManager.Ready = false;
					CommandManager.Error = true;
					CommandManager.CurrentState = EStateMachine::Faulted;
				}

			}


		}


		//set outputs
		out_Response = "ERROR";
		out_WaitingForAck = CommandManager.WaitingForACK;
		out_Ready = CommandManager.Ready;
		out_Error = CommandManager.Error;

		//set memory of reset bit state
		CommandManager.LastResetErrState = Onhigh_ResetError;

	}// continue waiting for active frame
	else { CommandManager.CommandManager_UpdateCounter = CommandManager.CommandManager_UpdateCounter + 1; }



}

void UIVILO_COMMS::CommandInputRequest(EMailBoxType MailBoxType, FString TargetID, float heading, float GimbalPitch, float height, float longitude, float latitude, float speed, bool isTurnClockwise)
{

	///type of COMMAND to push straight into COMMANDMANAGER, or user can clock it through manually
	//DEBUG CODE - NODE IS NOT UPDATING PRE-EXISTING TEST VALUES
	AssetReqCard.GimbalPitch = GimbalPitch;
	AssetReqCard.longitude = longitude;
	AssetReqCard.latitude = latitude;
	AssetReqCard.heading = heading;
	AssetReqCard.altitude = height;
	AssetReqCard.speed = speed;
	AssetReqCard.isClockWiseTurn = isTurnClockwise;
	AssetReqCard.NameOfAsset = TargetID;
	AssetReqCard.MailBoxType = MailBoxType;
	return;

}

//check mailbox
FString UIVILO_COMMS::MailBox_Check(EMailBoxType MailBoxFormat, FMailBoxInfo MailBoxInfoCard, FString MyID_internal)
{
	//UE_LOG(LogTemp, Error, TEXT("%s checking mail"), *MyID)

	if (MailBoxFormat == EMailBoxType::NoMailBox)
	{
		return "";
	}

	if (MailBoxFormat == EMailBoxType::FileLocation)
	{
		FString RootBox = MailBoxInfoCard.RootFolder + MyID_internal + MailBoxInfoCard.TreeLevel;
		FString InBoxe = RootBox + MailBoxInfoCard.InBox;
		FString MessageFile = InBoxe + MailBoxInfoCard.CommandFile;
		//check exists
		//IPlatformFile& PlatformFile = FPlatformFileManager::Get().GetPlatformFile();

		if (PlatformFile.FileExists(*MessageFile))
		{
			//UE_LOG(LogTemp, Error, TEXT("%s checking Mailbox message found"), *MyID)
			FString InputMessage;
			if (!FFileHelper::LoadFileToString(InputMessage, *MessageFile))
			{
				UE_LOG(LogTemp, Error, TEXT("%s ERROR check Mailbox message found but could not open"), *MyID)
					return "";
			}
			//delete folder then recreate to delete file
			if (!PlatformFile.DeleteDirectoryRecursively(*InBoxe))
			{
				UE_LOG(LogTemp, Error, TEXT("%s ERROR check Mailbox message can not delete folder after reading"), *MyID_internal)
					return "";
			}
			//recreate inbox folder
			if (!PlatformFile.CreateDirectoryTree(*InBoxe))
			{
				UE_LOG(LogTemp, Error, TEXT("%s ERROR check Mailbox message can not recreate folder after reading"), *MyID_internal)
					return "";
			}

			return InputMessage;
		}

		return "";
	}

	if (MailBoxFormat == EMailBoxType::LocalClipBoard)
	{
		return GetString_OSClipboard();
	}

	UE_LOG(LogTemp, Error, TEXT("%s CHECK ERROR, MAILBOX FORMAT IN DEVELOPMENT"), *MyID_internal)
		return "ERROR MAILBOX IN DEVELOPMENT";
}

//send message 
bool UIVILO_COMMS::MailBox_Send(EMailBoxType MailBoxFormat, FMailBoxInfo MailBoxInfoCard, FString TargetID, FString MessageBody)
{
	if (MailBoxFormat == EMailBoxType::NoMailBox)
	{
		return true;
	}

	if (MailBoxFormat == EMailBoxType::LocalClipBoard)
	{
		SendString_OSClipboard(MessageBody);
		return true;
	}

	if (MailBoxFormat == EMailBoxType::FileLocation)
	{
		//create directory string for ID in box
		FString RootBox = MailBoxInfoCard.RootFolder + TargetID + MailBoxInfoCard.TreeLevel;
		FString InBoxe = RootBox + MailBoxInfoCard.InBox;
		FString MessageFile = InBoxe + MailBoxInfoCard.CommandFile;
		//check exists
		//IPlatformFile& PlatformFile = FPlatformFileManager::Get().GetPlatformFile();
		if (!PlatformFile.DirectoryExists(*InBoxe))
		{
			UE_LOG(LogTemp, Error, TEXT("%s inbox doesnt exist, failed to send to : %s"), *TargetID, *InBoxe)
				return false;
		}

		if (PlatformFile.FileExists(*MessageFile))
		{
			UE_LOG(LogTemp, Error, TEXT("%s Mailbox full!!: %s"), *TargetID, *MessageFile)
				return false;
		}

		if (!FFileHelper::SaveStringToFile(MessageBody, *MessageFile))
		{
			UE_LOG(LogTemp, Error, TEXT("%s Mailbox error trying to save file!!: %s"), *TargetID, *MessageFile)
				return false;
		}

		return true;
	}

	UE_LOG(LogTemp, Error, TEXT("SEND ERROR, MAILBOX FORMAT IN DEVELOPMENT"))
		return false;
}

//check if folders exist
bool UIVILO_COMMS::PrepareMyFolders(FMailBoxInfo MailBoxInfoCard, FString MyID_Internal)
{
	UE_LOG(LogTemp, Error, TEXT("%s mailbox generation started... "), *MyID_Internal)

		//IPlatformFile& PlatformFile = FPlatformFileManager::Get().GetPlatformFile();
		// check root folder
		if ((MailBox_InfoCard.RootFolder.StartsWith("c")) || (MailBox_InfoCard.RootFolder.StartsWith("C")))
		{
			UE_LOG(LogTemp, Error, TEXT("%s mailbox illegal system drive for  %s, FAIL!!"), *MyID_Internal, *MailBox_InfoCard.RootFolder)
				return false;
		}
	if ((MailBox_InfoCard.RootFolder.StartsWith("d")) || (MailBox_InfoCard.RootFolder.StartsWith("D")))
	{
		UE_LOG(LogTemp, Error, TEXT("%s mailbox illegal system drive for  %s, FAIL!!"), *MyID_Internal, *MailBox_InfoCard.RootFolder)
			return false;
	}

	//check if main IVILO folder exists
	if (!PlatformFile.DirectoryExists(*MailBox_InfoCard.RootFolder))
	{
		UE_LOG(LogTemp, Error, TEXT("%s Checking top-level mailbox file location %s, FAIL!!"), *MyID_Internal, *MailBox_InfoCard.RootFolder)
			return false;
	}
	else
	{
		//UE_LOG(LogTemp, Error, TEXT("%s Checking mailbox file location %s, pass"), *MyID, *MailBox_InfoCard.RootFolder)
	}

	//check In Box directory exists for asset - delete if so
	FString TempPath = MailBox_InfoCard.RootFolder + MyID_Internal + MailBox_InfoCard.TreeLevel;
	if (PlatformFile.DirectoryExists(*TempPath))
	{
		//UE_LOG(LogTemp, Error, TEXT("%s re-initialising folder %s!!"), *MyID, *TempPath)
		if (!PlatformFile.DeleteDirectoryRecursively(*TempPath))
		{
			UE_LOG(LogTemp, Error, TEXT("%s mailbox error could not re-initialise (Delete) folder %s!! FAIL"), *MyID_Internal, *TempPath)
				return false;
		}
	}
	//make in-box for asset
	FString Boxes = TempPath + MailBox_InfoCard.InBox;
	if (!PlatformFile.CreateDirectoryTree(*Boxes))
	{
		UE_LOG(LogTemp, Error, TEXT("%s mailbox error could not create INBOX %s!! FAIL"), *MyID_Internal, *Boxes)
			return false;
	}
	Boxes = TempPath + MailBox_InfoCard.OutBox;
	//make OUT box now
	if (!PlatformFile.CreateDirectoryTree(*Boxes))
	{
		UE_LOG(LogTemp, Error, TEXT("%s mailbox error could not create OUTBOX  %s!! FAIL"), *MyID_Internal, *Boxes)
			return false;
	}
	//make tick/tock boxes
	FString Ticks = Boxes + MailBox_InfoCard.TickFolder;
	if (!PlatformFile.CreateDirectoryTree(*Ticks))
	{
		UE_LOG(LogTemp, Error, TEXT("%s mailbox error could not create Ticks folder  %s!! FAIL"), *MyID_Internal, *Ticks)
			return false;
	}
	Ticks = Boxes + MailBox_InfoCard.TockFolder;
	if (!PlatformFile.CreateDirectoryTree(*Ticks))
	{
		UE_LOG(LogTemp, Error, TEXT("%s mailbox error could not create Ticks folder  %s!! FAIL"), *MyID_Internal, *Ticks)
			return false;
	}



	Boxes = TempPath + MailBox_InfoCard.PictureBox;
	//make picture box
	if (!PlatformFile.CreateDirectoryTree(*Boxes))
	{
		UE_LOG(LogTemp, Error, TEXT("%s mailbox error could not create PictureBox  %s!! FAIL"), *MyID_Internal, *Boxes)
			return false;
	}

	UE_LOG(LogTemp, Error, TEXT("%s Mailbox generated"), *MyID_Internal)


		return true;
}

//saves on file system the current status/position using 2-file hand-over system
bool UIVILO_COMMS::SaveLivePositionStatus(FAssetRequestCard PositionCard, FMailBoxInfo InputMailBoxDetailCard)
{
	//uses TWO FILE SYSTEM - whereby this function saves up to 2 position files
	//and receiver will delete the oldest
		//give time for things to initialise
	if (GetWorld()->GetRealTimeSeconds() < 2)
	{
		return false;
	}

	//if (CommandHandler_FrameDivider < 0 || CommandHandler_FrameDivider >100)
	//{
	//	CommandHandler_FrameDivider = 30;
	//}
	//if (CommandManager.PositionWriter_UpdateCounter >= CommandHandler_FrameDivider + RandInt)
	//{
	//	RandInt = FMath::RandRange(0, CommonSet.RandomRange);//update random addition to update rate
	//	CommandManager.PositionWriter_UpdateCounter = 0;
	//}
	//else
	//{
	//	CommandManager.PositionWriter_UpdateCounter = CommandManager.PositionWriter_UpdateCounter + 1;
	//}



	//make sure asset card has been populated
	if (PositionCard.NameOfAsset == "" || isCommFoldersReady == false)
	{
		//UE_LOG(LogTemp, Error, TEXT("%s update position bad input conditions"), *PositionCard.NameOfAsset)
		return false;
	}

	bool isTickFileFound = false;
	bool isTockFileFound = false;
	FString TempPath = InputMailBoxDetailCard.RootFolder + PositionCard.NameOfAsset + MailBox_InfoCard.TreeLevel + InputMailBoxDetailCard.OutBox;
	FString TickFile = TempPath + InputMailBoxDetailCard.TickFolder + InputMailBoxDetailCard.Tick;
	FString TockFile = TempPath + InputMailBoxDetailCard.TockFolder + InputMailBoxDetailCard.Tock;
	//check if files exist
	isTickFileFound = PlatformFile.FileExists(*TickFile);
	isTockFileFound = PlatformFile.FileExists(*TockFile);
	//UE_LOG(LogTemp, Error, TEXT("tickfile %s"), *TickFile);
	//UE_LOG(LogTemp, Error, TEXT("tickfile %s"), *TockFile);
	//UE_LOG(LogTemp, Error, TEXT("isTickFileFound %s"), (isTickFileFound ? TEXT("True") : TEXT("False")));
	//UE_LOG(LogTemp, Error, TEXT("isTockFileFound %s"), (isTockFileFound ? TEXT("True") : TEXT("False")));

	//cannot update position if receiver not consuming input
	if (isTickFileFound && isTockFileFound)
	{
		return false;
	}

	//asset card values should have been updated by blueprint input command card
	FString OutputString = GenerateCommandString(EUnitCommands::ChangeDroneLatLongHeight, CommandManager.PositionWriter_Ticket, PositionCard.NameOfAsset);
	CommandManager.PositionWriter_Ticket = CommandManager.PositionWriter_Ticket + 1;//increment ticket number
	//save missing two-part file system
	if (!isTickFileFound)
	{
		if (!FFileHelper::SaveStringToFile(OutputString, *TickFile))
		{
			UE_LOG(LogTemp, Error, TEXT("%s ERROR sending tick position to : %s"), *PositionCard.NameOfAsset, *TickFile)
				return true;
		}
		//	UE_LOG(LogTemp, Error, TEXT("%s output TICK %s"), *PositionCard.NameOfAsset,*OutputString);

	}
	else if (!isTockFileFound)
	{
		if (!FFileHelper::SaveStringToFile(OutputString, *TockFile))
		{
			UE_LOG(LogTemp, Error, TEXT("%s ERROR sending tock position to : %s"), *PositionCard.NameOfAsset, *TockFile)
				return true;
		}
		//	UE_LOG(LogTemp, Error, TEXT("%s output TOCK %s"), *PositionCard.NameOfAsset, *OutputString);
	}
	/*else
	{
		CommandManager.PositionWriter_UpdateCounter = CommandManager.PositionWriter_UpdateCounter + 1;
	}*/
	return false;
}

//read and deletes files from 2 file passover system
void UIVILO_COMMS::ReadLivePositionStatus(FString AssetID, bool &PositionUpdated, float & latitude, float & longitude, float & heading, float &height, float & gimbalpitch, float & speed, int32 & LastUpdate_sec, FString &parsedAssetID, bool & isError)
{
	//give time for things to initialise
	if (GetWorld()->GetRealTimeSeconds() < 2)
	{
		return;
	}

	if (CommandHandler_FrameDivider < 0 || CommandHandler_FrameDivider >100)
	{
		CommandHandler_FrameDivider = 30;
	}

	LastUpdate_sec = GetWorld()->GetRealTimeSeconds() - CommandManager.PositionUpdater_LastUpdate;
	PositionUpdated = false;

	if (CommandManager.PositionUpdater_UpdateCounter >= CommandHandler_FrameDivider + RandInt)
	{
		RandInt = FMath::RandRange(0, CommonSet.RandomRange);//update random addition to update rate
		CommandManager.PositionUpdater_UpdateCounter = 0;

		//make sure asset card has been populated
		if (AssetID == "")
		{
			return;// false;
		}

		IPlatformFile& stupidPlatformFile = FPlatformFileManager::Get().GetPlatformFile();


		bool isTickFileFound = false;
		bool isTockFileFound = false;
		FString RootPath = MailBox_InfoCard.RootFolder + AssetID + MailBox_InfoCard.TreeLevel + MailBox_InfoCard.OutBox;

		FString TickFile = RootPath + MailBox_InfoCard.TickFolder + MailBox_InfoCard.Tick;
		FString TockFile = RootPath + MailBox_InfoCard.TockFolder + MailBox_InfoCard.Tock;
		FString TickFolder = RootPath + MailBox_InfoCard.TickFolder;
		FString TockFolder = RootPath + MailBox_InfoCard.TockFolder;

		//check if files exist
		isTickFileFound = stupidPlatformFile.FileExists(*TickFile);
		isTockFileFound = stupidPlatformFile.FileExists(*TockFile);
		//alternate what file we use as position
		FString InputMessage = CommonSet.EMPTY;
		if (isTickFileFound && isTockFileFound)//only read if both files available
		{
			//FDateTime FDateTimeTick = stupidPlatformFile.GetTimeStamp(*TickFile);
			//FDateTime FDateTimeTock = stupidPlatformFile.GetTimeStamp(*TockFile);

			//stupidPlatformFile.GetTimeStampPair(*TickFile, *TockFile, FDateTimeTick, FDateTimeTock);
			//int64 Tick_ticks = FDateTimeTick.GetTicks();
			//int64 Tock_ticks = FDateTimeTock.GetTicks();

			//trying to do with clock to improve resolution - obviously will break every hour!
			//int64 Tick_ticks32 = (FDateTimeTick.GetMinute()*60000)+ (FDateTimeTick.GetSecond() * 1000)+ FDateTimeTick.GetMillisecond();
			//int64 Tock_ticks32 = (FDateTimeTock.GetMinute()* 60000) + (FDateTimeTock.GetSecond() * 1000)+ FDateTimeTock.GetMillisecond();

			//if ticktime of one file is GREATER than another it must be newer
			//if (Tick_ticks < Tock_ticks)
			//{
			//	isExpectingTick = true;//smaller number - must be older
			//}
			//else
			//{
			//	isExpectingTick = false;
			//}

			if (isExpectingTick)
			{
				//try to load file
				if (!FFileHelper::LoadFileToString(InputMessage, *TickFile))
				{
					UE_LOG(LogTemp, Error, TEXT("ERROR loading live status/position file of asset : %s "), *TickFile)
						return;// false;
				}
				//now delete folder (cant delete files )
				if (!stupidPlatformFile.DeleteDirectoryRecursively(*TickFolder))
				{
					UE_LOG(LogTemp, Error, TEXT("%s mailbox error could not re-initialise (Delete) folder %s!! FAIL"), *AssetID, *TickFolder)
						return;// false;
				}
				//UE_LOG(LogTemp, Error, TEXT("%s TICK read position : %s"), *AssetID, *InputMessage)
				//return;// true;
			}
			else
			{
				//try to load file
				if (!FFileHelper::LoadFileToString(InputMessage, *TockFile))
				{
					UE_LOG(LogTemp, Error, TEXT("ERROR loading live status/position file of asset : %s"), *TockFile)
						return;//  false;
				}
				//now delete folder (cant delete files )
				if (!stupidPlatformFile.DeleteDirectoryRecursively(*TockFolder))
				{
					UE_LOG(LogTemp, Error, TEXT("%s mailbox error could not re-initialise (Delete) folder %s!! FAIL"), *AssetID, *TockFolder)
						return;//  false;
				}
				//isExpectingTick = true;
				//UE_LOG(LogTemp, Error, TEXT("%s TOCK read position : %s"), *AssetID, *InputMessage)
			//	return;// true;
			}
			isExpectingTick = !isExpectingTick;//alternate position sync
			//input positions found
			if (InputMessage != CommonSet.EMPTY)
			{
				//initialise info cards 
				FAssetInfoCard NewPositionsCard;
				FParseInputMessage ParseResult = ParseInputMsg(InputMessage, AssetID, NewPositionsCard);

				if (ParseResult.isForMe == true && ParseResult.isError == false)
				{
					//filter out syn errors - needs work
					if (CommandManager.PositionReader_Incomingticker > ParseResult.ParsedTicket)
					{
						UE_LOG(LogTemp, Error, TEXT("%s bad sync, ignoring position: %s"), *AssetID, *InputMessage)
							CommandManager.PositionReader_Incomingticker = ParseResult.ParsedTicket;
						return;
					}

					if (isExpectingTick)
					{
						//	UE_LOG(LogTemp, Error, TEXT("%s TICK in : %s"), *AssetID, *InputMessage)
					}
					else
					{
						//	UE_LOG(LogTemp, Error, TEXT("%s TOCK in : %s"), *AssetID, *InputMessage)
					}


					CommandManager.PositionReader_Incomingticker = ParseResult.ParsedTicket;

					//update outputs
					latitude = NewPositionsCard.latitude;
					longitude = NewPositionsCard.longitude;
					heading = NewPositionsCard.heading;
					height = NewPositionsCard.altitude;
					gimbalpitch = NewPositionsCard.GimbalPitch;
					speed = NewPositionsCard.speed;
					parsedAssetID = ParseResult.CallerID;
					CommandManager.PositionUpdater_LastUpdate = GetWorld()->GetRealTimeSeconds();

					isError = false;
					PositionUpdated = true;
					return;
				}

				//any other result is an error
				isError = false;
				UE_LOG(LogTemp, Error, TEXT("%s Reading Position unknown failure to parse: %s"), *AssetID, *InputMessage);
				return;
			}
		}

	}
	else
	{
		CommandManager.PositionUpdater_UpdateCounter = CommandManager.PositionUpdater_UpdateCounter + 1;
	}

	return;// false;
}

//takes image and saves as bmp
BOOL WINAPI TakeImageSaveBitmap(WCHAR *wPath, int32 iGrab_XShift)
{
	//https://stackoverflow.com/questions/3291167/how-can-i-take-a-screenshot-in-a-windows-application

	BITMAPFILEHEADER bfHeader;
	BITMAPINFOHEADER biHeader;
	BITMAPINFO bInfo;
	HGDIOBJ hTempBitmap;
	HBITMAP hBitmap;
	BITMAP bAllDesktops;
	HDC hDC, hMemDC;
	LONG lWidth, lHeight;
	BYTE *bBits = NULL;
	HANDLE hHeap = GetProcessHeap();
	DWORD cbBits, dwWritten = 0;
	HANDLE hFile;
	INT x = GetSystemMetrics(SM_XVIRTUALSCREEN);
	INT y = GetSystemMetrics(SM_YVIRTUALSCREEN);

	ZeroMemory(&bfHeader, sizeof(BITMAPFILEHEADER));
	ZeroMemory(&biHeader, sizeof(BITMAPINFOHEADER));
	ZeroMemory(&bInfo, sizeof(BITMAPINFO));
	ZeroMemory(&bAllDesktops, sizeof(BITMAP));

	hDC = GetDC(NULL);
	hTempBitmap = GetCurrentObject(hDC, OBJ_BITMAP);
	GetObjectW(hTempBitmap, sizeof(BITMAP), &bAllDesktops);

	lWidth = 900;//bAllDesktops.bmWidth;
	lHeight = 1200;//bAllDesktops.bmHeight;

	DeleteObject(hTempBitmap);

	bfHeader.bfType = (WORD)('B' | ('M' << 8));
	bfHeader.bfOffBits = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER);
	biHeader.biSize = sizeof(BITMAPINFOHEADER);
	biHeader.biBitCount = 24;
	biHeader.biCompression = BI_RGB;
	biHeader.biPlanes = 1;
	biHeader.biWidth = lWidth;
	biHeader.biHeight = lHeight;

	bInfo.bmiHeader = biHeader;

	cbBits = (((24 * lWidth + 31)&~31) / 8) * lHeight;

	hMemDC = CreateCompatibleDC(hDC);
	hBitmap = CreateDIBSection(hDC, &bInfo, DIB_RGB_COLORS, (VOID **)&bBits, NULL, 0);
	SelectObject(hMemDC, hBitmap);
	BitBlt(hMemDC, 0, 0, lWidth, lHeight, hDC, x + iGrab_XShift, y, SRCCOPY);


	hFile = CreateFileW(wPath, GENERIC_WRITE | GENERIC_READ, 0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
	WriteFile(hFile, &bfHeader, sizeof(BITMAPFILEHEADER), &dwWritten, NULL);
	WriteFile(hFile, &biHeader, sizeof(BITMAPINFOHEADER), &dwWritten, NULL);
	WriteFile(hFile, bBits, cbBits, &dwWritten, NULL);

	CloseHandle(hFile);

	DeleteDC(hMemDC);
	ReleaseDC(NULL, hDC);
	DeleteObject(hBitmap);

	return TRUE;
}

//// Fill out your copyright notice in the Description page of Project Settings.
//
//
//#include "IVILO_COMMS.h"
//
//// Sets default values for this component's properties
//UIVILO_COMMS::UIVILO_COMMS()
//{
//	// Set this component to be initialized when the game starts, and to be ticked every frame.  You can turn these features
//	// off to improve performance if you don't need them.
//	PrimaryComponentTick.bCanEverTick = true;
//
//	// ...
//}
//
//
//// Called when the game starts
//void UIVILO_COMMS::BeginPlay()
//{
//	Super::BeginPlay();
//
//	// ...
//	
//}
//
//
//// Called every frame
//void UIVILO_COMMS::TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction)
//{
//	Super::TickComponent(DeltaTime, TickType, ThisTickFunction);
//
//	// ...
//}
//
