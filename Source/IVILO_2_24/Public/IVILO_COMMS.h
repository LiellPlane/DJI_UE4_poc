// Fill out your copyright notice in the Description page of Project Settings.

#pragma once
#define WIN32_LEAN_AND_MEAN
#include "Windows/MinWindows.h"
#include <iostream>
#include <fstream>
#include <stdio.h>
#include "CoreMinimal.h"
#include "Containers/UnrealString.h"
#include "Components/ActorComponent.h"
#include "HAL/PlatformFilemanager.h"
#include "GenericPlatform/GenericPlatformFile.h"
#include "IVILO_COMMS.generated.h"


UENUM()
enum class EStateMachine : uint8
{
	Reseting,
	Faulted,
	Ready,
	Idle,
	Executing,
	Waiting
};



UENUM(BlueprintType)
enum class EMailBoxType : uint8
{
	NoMailBox UMETA(DisplayName = "NoMailBox"),
	HTTPLocation UMETA(DisplayName = "HTTPLocation"),
	FileLocation UMETA(DisplayName = "FileLocation"),
	LocalClipBoard UMETA(DisplayName = "LocalClipBoard")
};



UENUM(BlueprintType)
enum class EUnitCommands : uint8
{
	ChangeDroneLatLongHeight UMETA(DisplayName = "ChangeDroneLatLongHeight"),
	ReportAllPosition UMETA(DisplayName = "ReportAllPosition"),
	ChangeGimbalPitch UMETA(DisplayName = "ChangeGimbalPitch"),
	StopMission UMETA(DisplayName = "StopMission"),
	DemoMission UMETA(DisplayName = "DemoMission"),
	DoNothing UMETA(DisplayName = "DoNothing"),
	FAIL UMETA(DisplayName = "FAIL")
};

USTRUCT()
struct FMailBoxInfo
{
	GENERATED_USTRUCT_BODY()
		FString RootFolder = "Z:/IVILOBOX/";
	FString InBox = "INBOX/";
	FString OutBox = "OUTBOX/";
	FString PictureBox = "PictureBox/";
	FString TreeLevel = "/";
	FString CommandFile = "Message.txt";
	FString Tick = "Tick.txt";
	FString Tock = "Tock.txt";
	FString TickFolder = "Tick/";
	FString TockFolder = "Tock/";
};
USTRUCT()
struct FAssetInfoCard
{
	GENERATED_USTRUCT_BODY()
		FString NameOfAsset = "";
	double LastUpdated = 0;
	float altitude = 0.0f;
	float longitude = 0.0f;
	float latitude = 0.0f;
	float heading = 0.0f;
	float GimbalPitch = 0.0f;
	bool isTurningClockwise = false;
	float speed = 0.0f;
	FString Status = "";
	FString UnParsedInputString = "";
	int32 TicketNumber = 0;
	EUnitCommands InputCommand = EUnitCommands::DoNothing;
	bool OnReq_IsPass;
	FString OnReq_ErrMessage;
	bool OnReq_AckFinished = false;
};

USTRUCT()
struct FAssetRequestCard
{
	GENERATED_USTRUCT_BODY()
		FString NameOfAsset = "";
	float altitude = -9999.1234f;
	float longitude = -9999.1234f;
	float latitude = -9999.1234f;
	float heading = -9999.1234f;;
	float GimbalPitch = -9999.1234f;
	float speed = -9999.1234f;
	bool isClockWiseTurn = true;
	EMailBoxType MailBoxType = EMailBoxType::NoMailBox;
};

USTRUCT()
struct FCommonSet
{
	GENERATED_USTRUCT_BODY()
		//this much match up with strings from DJI SDK
		int32 ReportAllPosition_DelimitedCount = 11;
	int32 ReqPosition_CountNeeded = 4;//if user uses request card - has he updated all necessary elements (alt, long,lat etc)
	FString ACK = "ACK";
	FString REQ = "REQ";
	FString Error = "ERROR";
	FString FAIL = "FAIL";
	FString PASS = "PASS";
	FString EMPTY = "EMPTY";
	FString IVILO = "IVILO";
	FString Delimiter = "*";
	FString ChangeGimbalPitch = "ChangeGimbalPitch";
	FString ChangeDroneLatLongHeight = "ChangeDroneLatLongHeight";
	FString TakeOff = "TakeOff";
	FString ReportAllPosition = "ReportAllPosition";
	FString DemoMission = "DemoMission";
	FString StopMission = "StopMission";
	FString DoNothing = "DoNothing";
	FString ColourRED = "R=1,f G=0.f B=0.f A=1.f";
	FString ColourORANGE = "R=1,f G=0.5f B=0.f A=1.f";
	FString ColourGREEN = "R=0,f G=1.f B=0.f A=1.f";
	int32 RandomRange = 5;

};

USTRUCT()
struct FParseInputMessage
{
	GENERATED_USTRUCT_BODY()
		FString CallerID = "";
	FString TargetID = "";
	FString FailureReason = "";
	EUnitCommands ParsedCommand = EUnitCommands::DoNothing;
	int32 ParsedTicket = 0;
	bool isAck_elseReq = false;
	bool isError = false;
	int32 DelimitsFound = 0;
	bool isForMe = false;
	bool isCommandPass = false;
};

USTRUCT()
struct FSentMessage
{
	GENERATED_USTRUCT_BODY()
		FString TargetID = "";
	EUnitCommands SentCommand = EUnitCommands::DoNothing;
	int32 SendTicket = 0;
	double StartedTime = 0;
};



USTRUCT()
struct FCommandFunction_Features
{
	GENERATED_USTRUCT_BODY()
		int32 CommandManager_UpdateCounter = 10;
	int32 RequestManager_UpdateCounter = 10;
	int32 PositionUpdater_UpdateCounter = 10;
	int32 PositionWriter_UpdateCounter = 10;
	int32 PositionUpdater_LastUpdate = 0;
	int32 PositionWriter_Ticket = 1;
	int32 PositionReader_Incomingticker = 0;
	bool Listener_Ready = true;
	bool Ready = true;
	bool WaitingForACK = false;
	bool Error = false;
	bool LastResetErrState = false;
	FString CurrentREQ = "";
	int32 CurrentTicketNumber = 0;
	EStateMachine CurrentState = EStateMachine::Reseting;
	//double TimeStarted = 0.0f;
	FString LastClipBoardMsg = "";
	FString Listener_LastClipBoardMsg = "";
};



UCLASS( ClassGroup=(Custom), meta=(BlueprintSpawnableComponent) )
class IVILO_2_24_API UIVILO_COMMS : public UActorComponent
{
	GENERATED_BODY()

public:



	// Sets default values for this component's properties
	UIVILO_COMMS();

protected:
	// Called when the game starts
	virtual void BeginPlay() override;

public:
	// Called every frame
	virtual void TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction) override;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Enum)
		EUnitCommands DroneCommandSelect = EUnitCommands::DoNothing;
	//screenshot update rate
	UPROPERTY(EditAnywhere, Category = "Setup")
		int32 FrameDivider = 10;
	//timeout before command module errors
	UPROPERTY(EditAnywhere, Category = "Setup")
		int32 CommandACK_TimeOut_Secs = 5;
	//response speed
	UPROPERTY(EditAnywhere, Category = "Setup")
		int32 CommandHandler_FrameDivider = 10;
	//position on screen to start screen shot
	UPROPERTY(EditAnywhere, Category = "Setup")
		int32 Grab_XShift = 1000;
	//ID of sender
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Setup")
		FString MyID = "IVILO";

	UPROPERTY(BlueprintReadWrite, Category = "Setup")
		EMailBoxType MyMailBoxType = EMailBoxType::NoMailBox;

	UPROPERTY(BlueprintReadWrite, Category = "Setup")
		bool isCommFoldersReady = false;

	UPROPERTY(BlueprintReadWrite, Category = "Setup")
		FLinearColor IVILO_SpotLightColour;

	UPROPERTY(BlueprintReadWrite, Category = "Setup")
		FLinearColor ASSET_SpotLightColour;

	UFUNCTION(BlueprintCallable, Category = "Main")
		void TakeDeskTopImage(FString &FilePath_out, bool &isOK_out);

	UFUNCTION(BlueprintCallable, Category = "Main")
		void GetAssetPositions(float &latitude, float &longitude, float &heading, float &height, float &gimbalpitch, float & speed, int32 &LastUpdate_sec, FString &AssetID, bool &isError);

	UFUNCTION(BlueprintCallable, Category = "Main")
		void ListenForReqCommand(bool & IsReady, bool& NewInputCommand, EUnitCommands & RequestedCommand, float &latitude, float &longitude, float &heading, float &height, float &gimbalpitch, float&Speed, bool&isTurnClockWise, int32 &LastUpdate_sec, FString &AssetID, bool &isError);

	UFUNCTION(BlueprintCallable, Category = "Main")
		void ReplyClear_ReqCommand(bool isPASS, FString ErrorMsg);

	UFUNCTION(BlueprintCallable, Category = "internal")
		void SendString_OSClipboard(FString InputString);

	UFUNCTION(BlueprintCallable, Category = "internal")
		FString GetString_OSClipboard();

	UFUNCTION(BlueprintCallable, Category = "internal")
		FString GenerateCommandString(EUnitCommands InputCommand, int32 TicketNumber, FString TargetID);

	FString GenerateACKString(FAssetInfoCard InputReqCard);


	//cant use custom structure as output
	//UFUNCTION(BlueprintCallable, Category = "Main")
	FParseInputMessage ParseInputMsg(FString InputCommandString, FString MyIDinternal, FAssetInfoCard&OutputInfoCard);

	FString ConvertEnumToString(EUnitCommands InputCommand);

	EUnitCommands ConvertStringToEnum(FString InputCommand);

	UFUNCTION(BlueprintCallable, Category = "Main")
		bool GenerateCommFolders(FString Input_MyID);

	UFUNCTION(BlueprintCallable, Category = "Main")
		void CommandHandler(EUnitCommands InputCommand, bool Onhigh_ResetError, FString &out_Response, bool &out_WaitingForAck, bool &out_Ready, bool &out_Error);

	UFUNCTION(BlueprintCallable, Category = "Main")
		void CommandInputRequest(EMailBoxType MailBoxType, FString TargetID, float heading, float GimbalPitch, float height, float longitude, float latitude, float speed, bool isTurnClockwise);

	FString MailBox_Check(EMailBoxType MailBoxFormat, FMailBoxInfo MailBoxInfoCard, FString MyID);
	bool MailBox_Send(EMailBoxType MailBoxFormat, FMailBoxInfo MailBoxInfoCard, FString TargetID, FString MessageBody);
	bool PrepareMyFolders(FMailBoxInfo MailBoxInfoCard, FString MyID);


	bool SaveLivePositionStatus(FAssetRequestCard PositionCard, FMailBoxInfo InputMailBoxDetailCard);

	UFUNCTION(BlueprintCallable, Category = "Main")
		void ReadLivePositionStatus(FString AssetID, bool &PositionUpdated, float & latitude, float & longitude, float & heading, float &height, float & gimbalpitch, float & speed, int32 & LastUpdate_sec, FString &parsedAssetID, bool & isError);

private:
	int32 UpdateCounter = 0;
	FCommandFunction_Features CommandManager;
	FCommonSet CommonSet;
	FSentMessage SentMessageReceipt;
	FAssetInfoCard AssetInfoCard;
	FAssetRequestCard AssetReqCard;
	FAssetInfoCard InputPosReq;
	FMailBoxInfo MailBox_InfoCard;
	int32 RandInt = 0;
	//Float as String With Precision!
	FORCEINLINE FString GetFloatAsStringWithPrecision(float TheFloat, int32 Precision, bool IncludeLeadingZero = true)
	{
		//Round to integral if have something like 1.9999 within precision
		float Rounded = roundf(TheFloat);
		if (FMath::Abs(TheFloat - Rounded) < FMath::Pow(10, -1 * Precision))
		{
			TheFloat = Rounded;
		}
		FNumberFormattingOptions NumberFormat;					//Text.h
		NumberFormat.MinimumIntegralDigits = (IncludeLeadingZero) ? 1 : 0;
		NumberFormat.MaximumIntegralDigits = 10000;
		NumberFormat.MinimumFractionalDigits = Precision;
		NumberFormat.MaximumFractionalDigits = Precision;
		return FText::AsNumber(TheFloat, &NumberFormat).ToString();
	}
	IPlatformFile& PlatformFile = FPlatformFileManager::Get().GetPlatformFile();
	bool isExpectingTick = true;
};

BOOL __stdcall TakeImageSaveBitmap(WCHAR * wPath, int32 iGrab_XShift);


//// Fill out your copyright notice in the Description page of Project Settings.
//
//#pragma once
//
//#include "CoreMinimal.h"
//#include "Components/ActorComponent.h"
//#include "IVILO_COMMS.generated.h"
//
//
//UCLASS( ClassGroup=(Custom), meta=(BlueprintSpawnableComponent) )
//class IVILO_2_24_API UIVILO_COMMS : public UActorComponent
//{
//	GENERATED_BODY()
//
//public:	
//	// Sets default values for this component's properties
//	UIVILO_COMMS();
//
//protected:
//	// Called when the game starts
//	virtual void BeginPlay() override;
//
//public:	
//	// Called every frame
//	virtual void TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction) override;
//
//		
//};
